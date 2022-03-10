use nalgebra::DMatrix;
use std::fs;

pub fn load_csv(filepath: &str) -> Result<DMatrix<f64>, &str> {
    fs::read_to_string(filepath)
        .and_then(|c| Ok(c.split("\n").map(|l| l.trim()).filter(|l| l.len() > 0).map(
            |l| l.split(",")
                .map(|i| i.parse::<f64>().unwrap())
                .collect()
            )
            .collect::<Vec<Vec<f64>>>()
        ))
        .and_then(|data_vec| Ok(DMatrix::from_iterator(  // is read column-by-column, hence transpose result
            data_vec[0].len(),
            data_vec.len(),
            data_vec.into_iter().flatten()
        )))
        .or(Err("could not read csv"))
}