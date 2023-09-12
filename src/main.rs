use cl3::device::CL_DEVICE_TYPE_GPU;
use opencl3::platform::{Platform, get_platforms};
use opencl3::context::Context;
use opencl3::command_queue::CommandQueue;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_COPY_HOST_PTR};
use opencl3::memory::{CL_MEM_WRITE_ONLY, CL_MEM_USE_HOST_PTR};
use opencl3::device::Device;
use opencl3::program::Program;
use opencl3::kernel::{Kernel, CL_KERNEL_ARG_TYPE_NAME, ExecuteKernel};
use opencl3::types::{cl_event, CL_BLOCKING, CL_NON_BLOCKING};
use core::ptr::*;
use core::ffi::c_void;
use opencl3::Result;

const MY_PROGRAM : &str = 
"__kernel void vadd(__global const int *A, __global const int *B, __global int* C, unsigned int n){
    unsigned int i = get_global_id(0); 
    if(i < n){ 
        C[i] = A[i] + B[i]; 
    } 
}";
fn main() -> Result<()> {
    let platforms = get_platforms()?;
    assert!(0 < platforms.len());
    let platform = &platforms[0];
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU)?;
    println!("CL_DEVICE_TYPE_GPU count: {}", device_ids.len());
    assert!(0 < device_ids.len());
    let dev_id = device_ids[0];
    let device = Device::new(dev_id);
    println!("Device Created");
    let context = Context::from_device(&device)?;
    println!("Context Created");
    let command_queue = unsafe {
        CommandQueue::create_with_properties(&context, dev_id, 0 as u64, 0)?
    };
    println!("Command Queue Created");
    let read_flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    let write_flags = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR;
    let mut vec_a : Vec<i32> = (0..1024).collect();
    let mut vec_b : Vec<i32> = (0..1024).map(|x| x * 2).collect();
    let mut vec_c : Vec<i32> = vec![0; 1024];
    let mut buff_a : Buffer<i32> = unsafe {
        Buffer::create(&context, read_flags, 1024, vec_a.as_mut_ptr() as *mut c_void)?
    };
    let mut buff_b : Buffer<i32> = unsafe {
        Buffer::create(&context, read_flags, 1024, vec_b.as_mut_ptr() as *mut c_void)?
    };
    let mut buff_c : Buffer<i32> = unsafe {
        Buffer::create(&context, write_flags, 1024, vec_c.as_mut_ptr() as *mut c_void)?
    };
    println!("Buffers Created");
    let program = Program::create_and_build_from_source(&context, MY_PROGRAM, "").unwrap();
    println!("Program Created");
    let build_status = program.get_build_status(dev_id)?;
    println!("build_status : {}", build_status);
    let log = program.get_build_log(dev_id)?;
    println!("log : {}", log);
    println!("num kernels : {}", program.get_num_kernels()?);
    let kernel = Kernel::create(&program, "vadd")?;
    let global_size : usize = 1024;
    println!("Kernel Created");
    unsafe {
        println!("type of arg 0 : {:?}", kernel.get_arg_type_name(0)?);
        println!("{:?}", kernel.set_arg(0, &buff_a)?);
        println!("{:?}", kernel.set_arg(1, &buff_b)?);
        println!("{:?}", kernel.set_arg(2, &buff_c)?);
        println!("{:?}", kernel.set_arg(3, &(1024 as u32))?);
    }
    println!("Arguments Set");
    let work_group_size : usize = 64;
    println!("Work group size : {}", work_group_size);
    println!("Function name : {}", kernel.function_name()?);
    println!("Number of args : {}", kernel.num_args()?);
    println!("Attributes : {}", kernel.attributes()?);
    let global_offset : usize = 0;
    let event = unsafe {
        command_queue.enqueue_nd_range_kernel(kernel.into(), 1, &global_offset, &global_size, &work_group_size, &[])?
    };
    let event2 = unsafe {
        command_queue.enqueue_read_buffer(&buff_c, opencl3::memory::CL_TRUE, 0, vec_c.as_mut_slice(), &[cl_event::from(event)])?
    };
    for i in &vec_c {
        if *i > 16 {
            break;
        }
        println!("{ }", i);
    }   
   Ok(())
}
