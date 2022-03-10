use nalgebra::DVector;
use super::{init::span_clusters, distance};

fn calculate_centers(clustering: &Vec<Vec<&DVector<f64>>>) -> Vec<DVector<f64>> {
    // calculate the centers of given clusters

    clustering.iter()
        .map(|cluster| {
            cluster.iter()
            .fold(
                DVector::from_iterator(784, (0..784).into_iter().map(|_| 0.)),
                |s, &v| s + v
            ) * (1. / (if cluster.len() > 0 {cluster.len() as f64} else {1.}))
        }).collect()
}

fn assign(v: &DVector<f64>, centers: Vec<(usize, &DVector<f64>)>) -> usize {
    // return the index of the center nearest to the given vector

    let distances: Vec<(&usize, f64)> = centers.iter()
        .map(|(i, c)| (i, distance(&v, c)))
        .collect();

    let min = distances.iter()
        .map(|(_, d)| d)
        .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    distances.iter()
        .filter(|(_, d)| d == min)
        .map(|(i, _)| **i)
        .next()
        .unwrap()
}

fn cluster<'a>(train: &'a Vec<DVector<f64>>, centers: &Vec<DVector<f64>>) -> Vec<Vec<&'a DVector<f64>>> {
    // build clusters around given centers, put each vector in train into the cluster it belongs to
    // i.e., into the cluster with the closest center

    train.iter()
        .fold(
            vec![vec![]; centers.len()],
            |c, v| {
                let i = assign(&v, centers.iter().enumerate().collect());
                c.into_iter()
                    .enumerate()
                    .map(|(j, d)| if i == j {
                        d.iter().chain([v].iter()).map(|a| *a).collect()
                    } else {
                        d
                    }).map(|a| a).collect()
            }
        )
}

fn eq_clustering(left: &Vec<Vec<&DVector<f64>>>, right: &Vec<Vec<&DVector<f64>>>) -> bool {
    // check if two clusterings are the same

    left.iter()
        .zip(right.iter())
        .map(|(a, b)| a.iter()
                .zip(b)
                .skip_while(|(v, w)| v == w)
                .next()
                .is_some()
        )
        .take_while(|b| !b)
        .next()
        .is_some()
}

pub fn deterministic_kmeans(train: Vec<DVector<f64>>, k: usize) -> Vec<Vec<DVector<f64>>> {
    // Returns a clustering of the given vectors in train with k clusters

    let mut centers;
    let mut clustering;

    clustering = cluster(&train, &span_clusters(&train, k));
    centers = calculate_centers(&clustering);

    loop {
        let old = clustering;
        clustering = cluster(&train, &centers);
        centers = calculate_centers(&clustering);

        if eq_clustering(&old, &clustering) {
            break;
        }
    }

    clustering.into_iter()
        .map(|cluster| cluster.into_iter()
            .map(|v| v.clone())
            .collect()
        )
        .collect()
}