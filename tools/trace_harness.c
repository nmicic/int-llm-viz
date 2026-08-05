/*
 * Copyright 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * trace_harness.c — exact Q16.48 inference trace generator + verifier
 * ===================================================================
 *
 * Includes the vendored, unmodified upstream runtime
 * (third_party/int-llm/microgpt_int.c) so the ORIGINAL integer code is
 * the oracle. Two passes over the committed model.mgw:
 *
 *   Pass A (pristine): load the model, run mgpt_generate_sample() twenty
 *     times exactly as the upstream main() does, record the sample
 *     strings and the final PRNG state. No instrumentation touches this
 *     pass; it is the upstream behavior, byte for byte.
 *
 *   Pass B (instrumented): reload the model (weights and rng.state come
 *     back from the file, so the PRNG stream restarts identically) and
 *     replay generation step by step. For every (token, position) step:
 *       - call the ORIGINAL static inference_forward() for logits,
 *       - run an instrumented copy of the same loop, built from the
 *         SAME static building blocks (linear_fwd, rmsnorm_fwd,
 *         fp_mul, ...), using a separate KV cache, recording every
 *         intermediate value,
 *       - compare the two logits vectors and the two KV cache rows
 *         element-for-element (int64 equality, no tolerance).
 *     Sampling (temperature scale, softmax, weighted choice) replays the
 *     upstream mgpt_generate_sample() logic with the shared PRNG.
 *
 * The run FAILS (non-zero exit) unless: every logits vector matches,
 * every KV row matches, all twenty instrumented sample strings equal
 * the pristine strings, and the final PRNG state of pass B equals the
 * final PRNG state of pass A.
 *
 * Output: a single JSON document on the path given by argv[2] with
 * every recorded int64 as an exact decimal string. Floating point is
 * never used (the harness inherits the integer-only runtime).
 */

#define MGPT_NO_MAIN
#define MGPT_NO_TRAIN
#include "../third_party/int-llm/microgpt_int.c"

#include <inttypes.h>

#define NUM_SAMPLES 20

/* ------------------------------------------------------------------ */
/*  Instrumented KV cache (separate from the runtime's inf_keys/vals)  */
/* ------------------------------------------------------------------ */
static fixed_t my_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static fixed_t my_vals[N_LAYER][BLOCK_SIZE][N_EMBD];

/* Per-step instrumentation record (single layer, N_LAYER == 1). */
typedef struct {
    fixed_t emb[N_EMBD];        /* wte[token] + wpe[pos], before any norm */
    fixed_t x0[N_EMBD];         /* after initial RMSNorm                  */
    fixed_t scale_init;         /* RMSNorm scale of the initial norm      */
    fixed_t xn_attn[N_EMBD];    /* pre-attention RMSNorm output           */
    fixed_t scale_attn;
    fixed_t q[N_EMBD], k[N_EMBD], v[N_EMBD];
    fixed_t attn_scores[N_HEAD][BLOCK_SIZE]; /* scaled q.k, pre-softmax   */
    fixed_t attn_weights[N_HEAD][BLOCK_SIZE];/* post-softmax              */
    fixed_t ao[N_EMBD];         /* concatenated head outputs              */
    fixed_t x_mid[N_EMBD];      /* Wo(ao) + residual                      */
    fixed_t xn_mlp[N_EMBD];     /* pre-MLP RMSNorm output                 */
    fixed_t scale_mlp;
    fixed_t h_pre[MLP_DIM];     /* fc1 output before ReLU                 */
    fixed_t x_out[N_EMBD];      /* fc2 output + residual                  */
    fixed_t logits[MAX_CHARS + 1];
} instr_rec_t;

/* Instrumented replica of the upstream inference_forward(): identical
 * arithmetic through the same static helpers, but writing to my_keys /
 * my_vals and recording intermediates. Any divergence from the original
 * is caught by the caller's exact comparison. */
static void instr_forward(int token_id, int pos, instr_rec_t *rec) {
    fixed_t x[N_EMBD], tmp[MLP_DIM > N_EMBD ? MLP_DIM : N_EMBD];
    fixed_t xr[N_EMBD], xn[N_EMBD];
    fixed_t q[N_EMBD], k[N_EMBD], v[N_EMBD];
    fixed_t ao[N_EMBD], al[BLOCK_SIZE];
    fixed_t xn_m[N_EMBD], h1[MLP_DIM];

    for (int i = 0; i < N_EMBD; i++)
        x[i] = wte[token_id * N_EMBD + i] + wpe[pos * N_EMBD + i];
    memcpy(rec->emb, x, sizeof(x));
    rec->scale_init = rmsnorm_fwd(x, N_EMBD, x, NULL);
    memcpy(rec->x0, x, sizeof(x));

    for (int li = 0; li < N_LAYER; li++) {
        memcpy(xr, x, sizeof(xr));

        rec->scale_attn = rmsnorm_fwd(x, N_EMBD, xn, NULL);
        memcpy(rec->xn_attn, xn, sizeof(xn));

        linear_fwd(xn, attn_wq[li], N_EMBD, N_EMBD, q);
        linear_fwd(xn, attn_wk[li], N_EMBD, N_EMBD, k);
        linear_fwd(xn, attn_wv[li], N_EMBD, N_EMBD, v);
        memcpy(rec->q, q, sizeof(q));
        memcpy(rec->k, k, sizeof(k));
        memcpy(rec->v, v, sizeof(v));
        memcpy(my_keys[li][pos], k, sizeof(k));
        memcpy(my_vals[li][pos], v, sizeof(v));

        int seq_len = pos + 1;
        fixed_t attn_scale = ATTN_SCALE;
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;
            for (int s = 0; s < seq_len; s++) {
                fixed_t dot = 0;
                for (int j = 0; j < HEAD_DIM; j++)
                    dot += fp_mul(q[hs + j], my_keys[li][s][hs + j]);
                al[s] = fp_mul(dot, attn_scale);
                rec->attn_scores[h][s] = al[s];
            }
            fixed_t mx = al[0];
            for (int s = 1; s < seq_len; s++)
                if (al[s] > mx) mx = al[s];
            fixed_t sm = 0;
            for (int s = 0; s < seq_len; s++) {
                al[s] = fp_safe_exp(al[s] - mx);
                sm += al[s];
            }
            if (sm == 0) sm = 1;
            fixed_t inv_sm = fp_div(FP_ONE, sm);
            for (int s = 0; s < seq_len; s++) {
                al[s] = fp_mul(al[s], inv_sm);
                rec->attn_weights[h][s] = al[s];
            }
            for (int j = 0; j < HEAD_DIM; j++) {
                fixed_t acc = 0;
                for (int s = 0; s < seq_len; s++)
                    acc += fp_mul(al[s], my_vals[li][s][hs + j]);
                ao[hs + j] = acc;
            }
        }
        memcpy(rec->ao, ao, sizeof(ao));

        linear_fwd(ao, attn_wo[li], N_EMBD, N_EMBD, tmp);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = tmp[i] + xr[i];
        memcpy(rec->x_mid, x, sizeof(x));

        memcpy(xr, x, sizeof(xr));
        rec->scale_mlp = rmsnorm_fwd(x, N_EMBD, xn_m, NULL);
        memcpy(rec->xn_mlp, xn_m, sizeof(xn_m));
        linear_fwd(xn_m, mlp_fc1[li], MLP_DIM, N_EMBD, h1);
        memcpy(rec->h_pre, h1, sizeof(h1));
        for (int i = 0; i < MLP_DIM; i++)
            h1[i] = h1[i] > 0 ? h1[i] : 0; /* ReLU */
        linear_fwd(h1, mlp_fc2[li], N_EMBD, MLP_DIM, tmp);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = tmp[i] + xr[i];
        memcpy(rec->x_out, x, sizeof(x));
    }

    linear_fwd(x, lm_head, vocab_size, N_EMBD, rec->logits);
}

/* Upstream weighted_choice() with the drawn value and total exposed.
 * Same arithmetic, same single fp_rng_uniform() draw. */
static int weighted_choice_rec(const fixed_t *w, int n,
                               fixed_t *r_out, fixed_t *total_out) {
    fixed_t total = 0;
    for (int i = 0; i < n; i++) total += w[i];
    *total_out = total;
    if (total == 0) { *r_out = 0; return 0; }
    fixed_t r = fp_mul(fp_rng_uniform(), total);
    *r_out = r;
    fixed_t cum = 0;
    for (int i = 0; i < n; i++) {
        cum += w[i];
        if (r < cum) return i;
    }
    return n - 1;
}

/* ------------------------------------------------------------------ */
/*  JSON helpers (exact decimal strings for every int64)               */
/* ------------------------------------------------------------------ */
static void emit_i64(FILE *f, fixed_t v)  { fprintf(f, "\"%" PRId64 "\"", (int64_t)v); }
static void emit_u64(FILE *f, unsigned long long v) { fprintf(f, "\"%llu\"", v); }

static void emit_arr(FILE *f, const fixed_t *v, int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', f);
        emit_i64(f, v[i]);
    }
    fputc(']', f);
}

/* FNV-1a 64 over the little-endian 8-byte image of each element, in
 * index order. Mirrored by the Python and JS parsers as a cross-check
 * that all three read the identical tensor bytes. */
static uint64_t fnv1a64(const fixed_t *v, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        uint64_t u = (uint64_t)v[i];
        for (int b = 0; b < 8; b++) {
            h ^= (u >> (8 * b)) & 0xFF;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.mgw> <trace.json>\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    fp_math_init();

    /* ---------------- Pass A: pristine upstream behavior ------------- */
    if (load_model_mgw(model_path) != 0) return 1;
    unsigned long long rng_initial = fp_rng_state;

    char pristine[NUM_SAMPLES][BLOCK_SIZE + 1];
    for (int si = 0; si < NUM_SAMPLES; si++)
        mgpt_generate_sample(pristine[si]);
    unsigned long long rng_final_pristine = fp_rng_state;

    /* ---------------- Pass B: instrumented replay -------------------- */
    if (load_model_mgw(model_path) != 0) return 1;
    if (fp_rng_state != rng_initial) {
        fprintf(stderr, "FAIL: reload did not restore rng state\n");
        return 1;
    }
    memset(inf_keys, 0, sizeof(inf_keys));
    memset(inf_vals, 0, sizeof(inf_vals));
    memset(my_keys, 0, sizeof(my_keys));
    memset(my_vals, 0, sizeof(my_vals));

    FILE *f = fopen(argv[2], "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }

    /* ---- meta ---- */
    fprintf(f, "{\n\"meta\":{");
    fprintf(f, "\"format\":\"int-llm-viz-trace-v1\",");
    fprintf(f, "\"config\":{\"n_embd\":%d,\"n_head\":%d,\"n_layer\":%d,"
               "\"head_dim\":%d,\"mlp_dim\":%d,\"block_size\":%d,"
               "\"vocab_size\":%d,\"bos\":%d},",
            N_EMBD, N_HEAD, N_LAYER, HEAD_DIM, MLP_DIM, BLOCK_SIZE,
            vocab_size, BOS);
    fprintf(f, "\"uchars\":\"");
    for (int i = 0; i < num_uchars; i++) fputc(uchars_arr[i], f);
    fprintf(f, "\",");
    fprintf(f, "\"fp_one\":"); emit_i64(f, FP_ONE); fputc(',', f);
    fprintf(f, "\"attn_scale\":"); emit_i64(f, ATTN_SCALE); fputc(',', f);
    {
        fixed_t temperature = FP_ONE / 2;
        fixed_t inv_t = fp_div(FP_ONE, temperature);
        fprintf(f, "\"temperature\":"); emit_i64(f, temperature); fputc(',', f);
        fprintf(f, "\"inv_temperature\":"); emit_i64(f, inv_t); fputc(',', f);
    }
    fprintf(f, "\"rng_state_initial\":"); emit_u64(f, rng_initial);
    fprintf(f, ",\"num_params\":%d", num_params);

    /* ---- tensor checksums from the C loader's own arrays ---- */
    {
        mgpt_tensor_t tab[MGPT_NUM_WEIGHTS];
        int nw = mgpt_tensor_table(tab);
        fprintf(f, ",\"tensor_fnv1a64\":{");
        for (int t = 0; t < nw; t++) {
            size_t ne = (size_t)tab[t].rows * tab[t].cols;
            if (t) fputc(',', f);
            fprintf(f, "\"%s\":", tab[t].name);
            emit_u64(f, fnv1a64(*tab[t].slot, ne));
        }
        fprintf(f, "}");
    }
    fprintf(f, "},\n");

    /* ---- samples ---- */
    long logit_mismatches = 0, kv_mismatches = 0, steps_compared = 0;
    int sample_mismatches = 0;
    char instr_text[NUM_SAMPLES][BLOCK_SIZE + 1];

    fprintf(f, "\"samples\":[\n");
    for (int si = 0; si < NUM_SAMPLES; si++) {
        if (si) fprintf(f, ",\n");

        int slen = 0, token_id = BOS;
        char *text = instr_text[si];
        fixed_t temperature = FP_ONE / 2;
        fixed_t inv_t = fp_div(FP_ONE, temperature);

        fprintf(f, "{\"index\":%d,\"steps\":[", si);
        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            /* Oracle: the ORIGINAL, unmodified inference_forward. */
            fixed_t logits_orig[MAX_CHARS + 1];
            inference_forward(token_id, pos, logits_orig);

            /* Instrumented replica with its own KV cache. */
            static instr_rec_t rec;
            instr_forward(token_id, pos, &rec);

            /* Exact comparison: logits and this position's KV rows. */
            int step_logit_bad = 0, step_kv_bad = 0;
            for (int i = 0; i < vocab_size; i++)
                if (logits_orig[i] != rec.logits[i]) step_logit_bad++;
            for (int i = 0; i < N_EMBD; i++) {
                if (my_keys[0][pos][i] != inf_keys[0][pos][i]) step_kv_bad++;
                if (my_vals[0][pos][i] != inf_vals[0][pos][i]) step_kv_bad++;
            }
            logit_mismatches += step_logit_bad;
            kv_mismatches += step_kv_bad;
            steps_compared++;

            /* Sampling exactly as upstream mgpt_generate_sample(),
             * driven by the ORACLE's logits vector. */
            fixed_t scaled[MAX_CHARS + 1], probs[MAX_CHARS + 1];
            for (int i = 0; i < vocab_size; i++)
                scaled[i] = fp_mul(logits_orig[i], inv_t);
            unsigned long long rng_before = fp_rng_state;
            softmax_fwd(scaled, vocab_size, probs);
            fixed_t r, total;
            int chosen = weighted_choice_rec(probs, vocab_size, &r, &total);

            /* ---- step JSON ---- */
            if (pos) fputc(',', f);
            fprintf(f, "\n{\"pos\":%d,\"token_in\":%d,", pos, token_id);
            fprintf(f, "\"scale_init\":");  emit_i64(f, rec.scale_init);
            fprintf(f, ",\"scale_attn\":"); emit_i64(f, rec.scale_attn);
            fprintf(f, ",\"scale_mlp\":");  emit_i64(f, rec.scale_mlp);
            fprintf(f, ",\"emb\":");     emit_arr(f, rec.emb, N_EMBD);
            fprintf(f, ",\"x0\":");      emit_arr(f, rec.x0, N_EMBD);
            fprintf(f, ",\"xn_attn\":"); emit_arr(f, rec.xn_attn, N_EMBD);
            fprintf(f, ",\"q\":");       emit_arr(f, rec.q, N_EMBD);
            fprintf(f, ",\"k\":");       emit_arr(f, rec.k, N_EMBD);
            fprintf(f, ",\"v\":");       emit_arr(f, rec.v, N_EMBD);
            fprintf(f, ",\"attn\":[");
            for (int h = 0; h < N_HEAD; h++) {
                if (h) fputc(',', f);
                fprintf(f, "{\"scores\":");
                emit_arr(f, rec.attn_scores[h], pos + 1);
                fprintf(f, ",\"weights\":");
                emit_arr(f, rec.attn_weights[h], pos + 1);
                fputc('}', f);
            }
            fprintf(f, "]");
            fprintf(f, ",\"ao\":");      emit_arr(f, rec.ao, N_EMBD);
            fprintf(f, ",\"x_mid\":");   emit_arr(f, rec.x_mid, N_EMBD);
            fprintf(f, ",\"xn_mlp\":");  emit_arr(f, rec.xn_mlp, N_EMBD);
            fprintf(f, ",\"h_pre\":");   emit_arr(f, rec.h_pre, MLP_DIM);
            fprintf(f, ",\"x_out\":");   emit_arr(f, rec.x_out, N_EMBD);
            fprintf(f, ",\"logits\":");  emit_arr(f, rec.logits, vocab_size);
            fprintf(f, ",\"logits_scaled\":"); emit_arr(f, scaled, vocab_size);
            fprintf(f, ",\"probs\":");   emit_arr(f, probs, vocab_size);
            fprintf(f, ",\"prob_total\":"); emit_i64(f, total);
            fprintf(f, ",\"rng_before\":"); emit_u64(f, rng_before);
            fprintf(f, ",\"r\":");       emit_i64(f, r);
            fprintf(f, ",\"chosen\":%d", chosen);
            fprintf(f, ",\"checks\":{\"logits_equal\":%s,\"kv_equal\":%s}}",
                    step_logit_bad ? "false" : "true",
                    step_kv_bad ? "false" : "true");

            token_id = chosen;
            if (token_id == BOS) break;
            if (token_id < num_uchars)
                text[slen++] = uchars_arr[token_id];
        }
        text[slen] = '\0';

        int match = strcmp(text, pristine[si]) == 0;
        if (!match) sample_mismatches++;
        fprintf(f, "],\"text\":\"%s\",\"pristine_text\":\"%s\","
                   "\"match_pristine\":%s}",
                text, pristine[si], match ? "true" : "false");

        /* Upstream clears its KV cache after each sample; mirror both. */
        memset(inf_keys, 0, sizeof(inf_keys));
        memset(inf_vals, 0, sizeof(inf_vals));
        memset(my_keys, 0, sizeof(my_keys));
        memset(my_vals, 0, sizeof(my_vals));
    }
    fprintf(f, "\n],\n");

    unsigned long long rng_final_instr = fp_rng_state;
    int rng_match = rng_final_instr == rng_final_pristine;

    fprintf(f, "\"verification\":{");
    fprintf(f, "\"steps_compared\":%ld,", steps_compared);
    fprintf(f, "\"logit_mismatches\":%ld,", logit_mismatches);
    fprintf(f, "\"kv_mismatches\":%ld,", kv_mismatches);
    fprintf(f, "\"sample_mismatches\":%d,", sample_mismatches);
    fprintf(f, "\"rng_final_pristine\":"); emit_u64(f, rng_final_pristine);
    fprintf(f, ",\"rng_final_instrumented\":"); emit_u64(f, rng_final_instr);
    fprintf(f, ",\"rng_final_match\":%s", rng_match ? "true" : "false");
    fprintf(f, ",\"pristine_samples\":[");
    for (int si = 0; si < NUM_SAMPLES; si++)
        fprintf(f, "%s\"%s\"", si ? "," : "", pristine[si]);
    fprintf(f, "]");
    int pass = !logit_mismatches && !kv_mismatches && !sample_mismatches
               && rng_match;
    fprintf(f, ",\"pass\":%s}", pass ? "true" : "false");
    fprintf(f, "\n}\n");
    fclose(f);

    fprintf(stderr,
            "trace: %ld steps, %ld logit mismatches, %ld kv mismatches, "
            "%d sample mismatches, rng %s => %s\n",
            steps_compared, logit_mismatches, kv_mismatches,
            sample_mismatches, rng_match ? "match" : "MISMATCH",
            pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
