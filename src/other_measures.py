"""Module providing functions to calculate the Shepard Goodness score, plot 
the Shepard diagram, and calculate the scalar which makes the area of the 
Shepard diagram square for each dimensionality reduction technique."""

from zadu.measures import *


def shepard(X, techniques):
    """Function that calculates the Shepard Goodness score."""
    correlations = []

    projections = [techniques['tsne'][0], techniques['umap'][0],
                   techniques['mds'][0], techniques['random'][0]]
    for projection in projections:
        # Compute the correlation between distances
        corr = spearman_rho.measure(X, projection)['spearman_rho']
        correlations.append(corr)

    return correlations


def trustworthiness_and_continuity(X, techniques):
    """Function that calculates the trustworthiness and continuity."""
    results_trustworthiness = []
    results_continuity = []

    projections = [techniques['tsne'][0], techniques['umap'][0],
                   techniques['mds'][0], techniques['random'][0]]
    for projection in projections:
        # Compute the trustworthiness and continuity
        tmp = trustworthiness_continuity.measure(X, projection)
        trustworthiness = tmp['trustworthiness']
        continuity = tmp['continuity']

        results_trustworthiness.append(trustworthiness)
        results_continuity.append(continuity)

    return results_trustworthiness, results_continuity


def mrre(X, techniques):
    """Function that calculates the MRRE."""
    results_false = []
    results_missing = []

    projections = [techniques['tsne'][0], techniques['umap'][0],
                   techniques['mds'][0], techniques['random'][0]]
    for projection in projections:
        # Compute the MRREs
        tmp = mean_relative_rank_error.measure(X, projection)
        false = tmp['mrre_false']
        missing = tmp['mrre_missing']

        results_false.append(false)
        results_missing.append(missing)

    return results_false, results_missing


def neighbor_hit(X, techniques):
    """Function that calculates the neighborhood hit."""
    hits = []

    projections = [techniques['tsne'][0], techniques['umap'][0],
                   techniques['mds'][0], techniques['random'][0]]
    for projection in projections:
        # Compute the neighborhood hits
        hit = neighborhood_hit.measure(
            X, projection)['neighborhood_hit']
        hits.append(hit)

    return hits


def steadiness_and_cohesiveness(X, techniques):
    """Function that calculates the steadiness and cohesiveness."""
    results_steadiness = []
    results_cohesiveness = []

    projections = [techniques['tsne'][0], techniques['umap'][0],
                   techniques['mds'][0], techniques['random'][0]]
    for projection in projections:
        # Compute the steadiness and cohesiveness
        tmp = steadiness_cohesiveness.measure(X, projection)
        steadiness = tmp['steadiness']
        cohesiveness = tmp['cohesiveness']

        results_steadiness.append(steadiness)
        results_cohesiveness.append(cohesiveness)

    return results_steadiness, results_cohesiveness


def kl(X, techniques):
    """Function that calculates the Kullback-Leibler Divergence."""
    divergences = []

    projections = [techniques['tsne'][0], techniques['umap'][0],
                   techniques['mds'][0], techniques['random'][0]]
    for projection in projections:
        # Compute the Kullback-Leibler Divergence
        divergence = kl_divergence.measure(
            X, projection)['kl_divergence']
        divergences.append(divergence)

    return divergences


def pearson_corr(X, techniques):
    """Function that calculates the pearson correlation."""
    correlations = []

    projections = [techniques['tsne'][0], techniques['umap'][0],
                   techniques['mds'][0], techniques['random'][0]]
    for projection in projections:
        # Compute the correlation between distances
        corr = pearson_r.measure(X, projection)['pearson_r']
        correlations.append(corr)

    return correlations
