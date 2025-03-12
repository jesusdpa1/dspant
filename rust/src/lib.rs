use pyo3::prelude::*;

#[pyfunction]
fn print_hello(name: &str) -> PyResult<String> {
    let message = format!("Hello, {}!", name);
    println!("{}", message);
    Ok(message)
}

#[pyfunction]
fn guess_the_number() -> PyResult<()> {
    // You'll implement the guessing game logic here
    Ok(())
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(print_hello, m)?)?;
    m.add_function(wrap_pyfunction!(guess_the_number, m)?)?;

    Ok(())
}
