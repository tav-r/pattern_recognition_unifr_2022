use nalgebra::DVector;

pub mod csv;
pub mod init;
pub mod cluster;
pub mod indices;

fn distance(v: &DVector<f64>, w: &DVector<f64>) -> f64 {
    let diff = v - w;

    diff.dot(&diff)
}

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