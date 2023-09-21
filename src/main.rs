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
    assert!(0 < device_ids.len());
    let dev_id = device_ids[0];
    let device = Device::new(dev_id);
    let context = Context::from_device(&device)?;
    let command_queue = unsafe {
        CommandQueue::create_with_properties(&context, dev_id, 0 as u64, 0)?
    };
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
    let program = Program::create_and_build_from_source(&context, MY_PROGRAM, "").unwrap();
    let build_status = program.get_build_status(dev_id)?;
    let log = program.get_build_log(dev_id)?;
    let kernel = Kernel::create(&program, "vadd")?;
    let global_size : usize = 1024;
    unsafe {
        kernel.set_arg(0, &buff_a)?;
        kernel.set_arg(1, &buff_b)?;
        kernel.set_arg(2, &buff_c)?;
        kernel.set_arg(3, &(1024 as u32))?;
    }
    let work_group_size : usize = 64;
    let global_offset : usize = 0;
    let event = unsafe {
        command_queue.enqueue_nd_range_kernel(kernel.get(), 1, &global_offset, &global_size, &work_group_size, &[]).unwrap()
    };
    println!("Kernel run");
    let event2 = unsafe {
        command_queue.enqueue_read_buffer(&buff_c, opencl3::memory::CL_TRUE, 0, vec_c.as_mut_slice(), &[event.get()])?
    };
    for i in &vec_c {
        if *i > 16 {
            break;
        }
        println!("{ }", i);
    }   
   Ok(())
}
