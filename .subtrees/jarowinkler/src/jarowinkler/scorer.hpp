#pragma once
#include "common.hpp"

static inline double jaro_similarity_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return jaro_winkler::jaro_similarity(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool JaroSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return scorer_init_f64<jaro_winkler::CachedJaroSimilarity>(self, str_count, str);
}

static inline double jaro_winkler_similarity_func(const RF_String& s1, const RF_String& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return jaro_winkler::jaro_winkler_similarity(first1, last1, first2, last2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return scorer_init_f64<jaro_winkler::CachedJaroWinklerSimilarity>(self, str_count, str, *(double*)(kwargs->context));
}
