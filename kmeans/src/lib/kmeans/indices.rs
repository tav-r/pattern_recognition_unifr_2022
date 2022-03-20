use nalgebra::DVector;
use super::{distance,calculate_centers};

fn single_linkage(c1: &Vec<&DVector<f64>>, c2: &Vec<&DVector<f64>>) -> f64 {
    c1.iter()
        .map(|v1|
            c2.iter()
            .map(|v2| distance(v1, v2))
        )
        .flatten()
        .min_by(|d1, d2| d1.partial_cmp(d2).unwrap())
        .unwrap()
}

fn diameter(cluster: &Vec<&DVector<f64>>) -> f64 {
    cluster.iter()
        .enumerate()
        .map(|(i, v1)| cluster.iter()
            .skip(i+1)
            .map(|v2| distance(v1, v2))
        )
        .flatten()
        .max_by(|d1, d2| d1.partial_cmp(d2).unwrap())
        .unwrap()
}

pub fn dunn_index(clustering: &Vec<Vec<&DVector<f64>>>) -> f64 {
    let max_diameter = clustering.iter()
        .map(|c| diameter(c))
        .max_by(|d1, d2| d1.partial_cmp(d2).unwrap())
        .unwrap();

    let min_single_linkage = clustering
        .iter()
        .enumerate()
        .map(|(i, c1)| clustering.iter()
            .skip(i+1)
            .map(|c2| single_linkage(c1, c2))
        ).flatten()
        .min_by(|d1, d2| d1.partial_cmp(d2).unwrap())
        .unwrap();

    min_single_linkage / max_diameter
}

pub fn davis_bouldin_index(clustering: &Vec<Vec<&DVector<f64>>>) -> f64 {
    let ms = calculate_centers(clustering);
    let ds: Vec<f64> = clustering.iter().zip(ms.iter())
        .map(|(c, m)|
            (1./c.len() as f64) * c.iter().map(|v| distance(v, m)).sum::<f64>()
        ).collect();

    let ris: Vec<f64> = ms.iter()
        .zip(ds.iter())
        .enumerate()
        .map(|(i, (m1, d1))|  // for each pair (m_i, d_i)...
            ms.iter()
                .zip(ds.iter())
                .enumerate()
                .filter(|(j, _)| *j != i)  // ...take pair (m_j, d_j) with j!=i...
                .map(|(_, (m2, d2))| 
                    (d1 + d2) / distance(m1, m2)  // ...calculate R_ij...
                ).max_by(|rij, rij_| rij.partial_cmp(rij_)  // ...and choose the maximum to get R_i
                .unwrap()
            )
            .unwrap()
        )
        .collect();

    ris.iter().sum::<f64>() / (ris.len() as f64)
}