fn main() {
    match oxicuda_runtime::get_device_count() {
        Ok(n) => println!("get_device_count() = {n}"),
        Err(e) => println!("get_device_count() error: {e:?}"),
    }
    match oxicuda_runtime::device::set_device(0) {
        Ok(()) => println!("set_device(0) = Ok"),
        Err(e) => println!("set_device(0) error: {e:?}"),
    }
    match oxicuda_runtime::cuda_malloc(1024) {
        Ok(p) => {
            println!("cuda_malloc(1024) = Ok({p:?})");
            let _ = oxicuda_runtime::cuda_free(p);
        }
        Err(e) => println!("cuda_malloc error: {e:?}"),
    }
}
