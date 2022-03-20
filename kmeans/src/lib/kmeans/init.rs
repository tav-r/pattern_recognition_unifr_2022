use nalgebra::DVector;
use super::distance;

fn max_min<'a>(s: &'a Vec<DVector<f64>>, m: &'a Vec<DVector<f64>>) -> &'a DVector<f64> {
    // get the vector in s which maximizes the minimal distance to any vector in m

    s.iter()
        .filter(|v| !m.contains(v))  // take only vectors that are not already in m
        .map(|v| {  // for each vector in s ...
            let d = m.iter()  // ... calculate distance to every vector in m ...
                .map(|w| distance(v, w))
                .min_by(|a, b| a.partial_cmp(b).unwrap())  // and choose the smallest distance
                .unwrap();

            (v, d)  // return minimal distance and index of vector in s that produced it
        })
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())  // select tuple with smallest second element (distance)
        .unwrap()
        .0  // select first element of tuple (vector)
}

pub fn span_clusters(
    train: &Vec<DVector<f64>>,
    n_clusters: usize
) -> Vec<DVector<f64>> {
    // spanning selection for initial clustering, used by deterministic K-Means

    (1..n_clusters).into_iter().fold(
        vec![&train.iter()  // start with central vector
            .skip(1)
            .fold(  // sum up all vectors
                train[0].clone(),
                |v, w| w + v
            ) * (1. / (train.len() as f64))  // divide each component "sum-vector" by number of vectors
        ],
        |v, _| {
            v.iter()
                .chain([max_min(&train, &v)].into_iter())
                .map(|v| v.clone())
                .collect()
        }
    )
}