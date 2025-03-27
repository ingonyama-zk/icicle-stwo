use std::cell::RefCell;
use std::time::Instant;

use nvtx::{range_pop, range_push};

thread_local! {
    pub static NVTX_STACK: RefCell<Vec<NvtxGuard>> = RefCell::new(Vec::new());
}

pub struct NvtxGuard {
    description: String,
    start: Instant,
}

impl NvtxGuard {
    pub fn new(description: &str, indent: usize) -> Self {
        let indented_desc = format!("{}{}", " ".repeat(indent), description);
        println!("Starting: {}", indented_desc);
        range_push!("{}", indented_desc.as_str());
        Self {
            description: indented_desc,
            start: Instant::now(),
        }
    }

    pub fn pop(self) {
        range_pop!();
        let duration = self.start.elapsed();
        println!("End of    {}: time: {:?}", self.description, duration);
    }
}

#[macro_export]
macro_rules! nvtx_timed {
    ($desc:expr) => {
        crate::utils2::NVTX_STACK.with(|stack| {
            let indent = stack.borrow().len();
            stack
                .borrow_mut()
                .push(crate::utils2::NvtxGuard::new($desc, indent));
        });
    };
}

#[macro_export]
macro_rules! nvtx_timed_pop {
    () => {
        crate::utils2::NVTX_STACK.with(|stack| {
            if let Some(guard) = stack.borrow_mut().pop() {
                guard.pop();
            }
        });
    };
}
