extern crate nalgebra as na;

use na::{DMatrix, DVector};
use std::{fs, error::Error, iter::Sum};

fn parse_csv(filepath: &str) -> Result<DMatrix<i64>, &str> {
    fs::read_to_string(filepath)
        .and_then(|c| Ok(c.split("\n").map(|l| l.trim()).filter(|l| l.len() > 0).map(
            |l| l.split(",")
                .map(|i| i.parse::<i64>().unwrap())
                .collect()
            )
            .collect::<Vec<Vec<i64>>>()
        ))
        .and_then(|data_vec| Ok(DMatrix::from_iterator(  // is read column-by-column, hence transpose result
            data_vec[0].len(),
            data_vec.len(),
            data_vec.into_iter().flatten()
        ).transpose()))
        .or(Err("could not read csv"))
}

// using quadratic euclidian distance since taking the square root again only CPU time
#[allow(dead_code)]
fn quad_euclidian_distance(left: &DVector<i64>, right: &DVector<i64>) -> i64 {
    Sum::sum(left.iter().zip(right.iter()).map(|(x, y)| (x - y) * (x - y)))
}

#[allow(dead_code)]
fn manhatten_distance(left: &DVector<i64>, right: &DVector<i64>) -> i64 {
    Sum::sum(left.iter().zip(right.iter()).map(|(x, y)| i64::abs(x - y)))
}

fn knn(
    model: DMatrix<i64>,
    test: DMatrix<i64>,
    values: Vec<i64>,
    distance_function: &dyn Fn(&DVector<i64>, &DVector<i64>) -> i64
) -> Vec<(usize, Vec<usize>)> {
    // calculate the distance to each row in the model for each test-data row
    let rows_distances: Vec<Vec<(usize, i64)>> = test.row_iter().map(|row| {
        // build a matrix with same size as model, where each row is the sample vector
        let test_vector = DVector::from_iterator(
            model.ncols(),
            row.into_iter().skip(1).map(|&n| n)  // skip the row with the digit
        );

        // calculate the distance between each row vector of the two matrices
        let mut distances: Vec<(usize, i64)> = model
            .row_iter()
            .enumerate()
            .map(|(i, r)| (i, distance_function(&r.transpose(), &test_vector)))
            .collect();

        // sort the vector by values
        distances.sort_by_key(|&(_, v)| v);

        distances
    }).collect();

    // find the K nearest for some Ks
    [1, 3, 5, 10, 15].into_iter().map(|k| {
        (k, rows_distances.iter().map(|distances| {
            // select the k nearest digits
            let k_nearest: Vec<i64> = distances
                .iter()
                .take(k)
                .map(|&(i, _)| values[i])
                .collect();

            // create 10 bins, one for each possible digit, each holding the number of occurrences of that digit
            let bincount: Vec<usize> = (0..10)
                .into_iter()
                .map(|i| k_nearest.iter().filter(|&&j| j == i).count())
                .collect();

            // take the digit that appears most often
            bincount
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap()
        }).collect()
    )}).collect()
}

// Assumes "test.csv" and "train.csv" to be in the working directory.
fn main() -> Result<(), Box<dyn Error>> {
    let data_matrix = parse_csv("train.csv")?;

    // load an parse data
    let values: Vec<i64> = data_matrix.columns(0, 1).iter().map(|v| *v).collect();  // digit
    let model = data_matrix.columns(1, data_matrix.ncols() - 1);  // digit pixels

    let test = parse_csv("test.csv")?;
    let test_values: Vec<i64> = test.columns(0, 1).iter().map(|v| *v).collect();

    // classify the data from the "test.csv" using the model data from "train.csv"
    let res: Vec<(usize, Vec<usize>)> = knn(model.into(), test, values, &quad_euclidian_distance);  // choose distance function here

    // count and report the misclassifications for each (k, digits)-pair in the resulting classification
    let errors: Vec<(usize, usize)> = res
        .iter()
        .map(|(k, vals)| (
            *k, vals.iter()
                .zip(test_values.iter())
                .filter(|(&x, &y)| (x as i64) != y)
                .count()))
            .collect();

    for (k, e) in errors {
        println!("{} errors in {} classifications with k={}", e, test_values.len(), k);
    }

    Ok(())
}