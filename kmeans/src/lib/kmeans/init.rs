use nalgebra::DVector;
use super::distance;

fn max_min<'a>(s: &'a Vec<DVector<f64>>, m: &'a Vec<DVector<f64>>) -> &'a DVector<f64> {
    // get index of vector in s which maximizes the minimal distance to any vector in m
    let mins: Vec<(&DVector<f64>, f64)> = s.iter()
        .enumerate()
        .filter(|(_, v)| !m.contains(v))  // filter out all vectors that are already in m
        .map(|(_, v)| {  // for each vector in s ...
            let d = m.iter()  // ... calculate distance to every vector in m ...
                .map(|w| distance(v, w))
                .min_by(|a, b| a.partial_cmp(b).unwrap())  // and choose the smallest distance
                .unwrap();

            (v, d)  // return minimal distance and index of vector in s that produced it
        }).collect();

    let min = mins.iter().map(|(_, d)| d).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    mins.iter()
        .filter(|(_, d)| d == min)
        .map(|(v, _)| v)
        .next()
        .unwrap()
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
            ) * (1.0 / (train.len() as f64))  // divide each component "sum-vector" by number of vectors
        ],
        |v, _| {
            v.iter()
                .chain([max_min(&train, &v)].into_iter())
                .map(|v| v.clone())
                .collect()
        }
    )
}