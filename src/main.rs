use cl3::device::CL_DEVICE_TYPE_GPU;
use opencl3::platform::Platform;
use opencl3::platform::get_platforms;
use opencl3::context::Context;
use opencl3::command_queue::CommandQueue;
use opencl3::memory::*;
use opencl3::device::Device;
use opencl3::program::Program;
use opencl3::kernel::Kernel;
use opencl3::types::cl_event;
use core::ptr::*;
use core::ffi::c_void;
//use std::env;

const MY_PROGRAM : &str = 
"__kernel void vadd(__global const int *A, __global const int *B, __global int* C, unsigned long n){
    int i = get_global_id(0); 
    if(i < n){ 
        C[i] = A[i] + B[i]; 
    } 
}";
fn main() {
    //env::set_var("RUST_BACKTRACE", "full");
    let platforms = get_platforms().unwrap();
    assert!(0 < platforms.len());
    let platform = &platforms[0];
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
    println!("CL_DEVICE_TYPE_GPU count: {}", device_ids.len());
    assert!(0 < device_ids.len());
    let dev_id = device_ids[0];
    let device = Device::new(dev_id);
    println!("Device Created");
    let context = Context::from_device(&device).unwrap();
    println!("Context Created");
    let command_queue = unsafe {
        CommandQueue::create_with_properties(&context, dev_id, 0 as u64, 0).unwrap()
    };
    println!("Command Queue Created");
    let read_flags = opencl3::memory::CL_MEM_HOST_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR;
    let write_flags = opencl3::memory::CL_MEM_READ_WRITE | opencl3::memory::CL_MEM_USE_HOST_PTR;
    let mut vec_a : Vec<i32> = (0..1024).collect();
    let mut vec_b : Vec<i32> = (0..1024).map(|x| x * 2).collect();
    let mut vec_c : Vec<i32> = vec![0; 1024];
    let mut buff_a : opencl3::memory::Buffer<i32> = unsafe {
        opencl3::memory::Buffer::create(&context, read_flags, 1024, vec_a.as_mut_ptr() as *mut c_void).unwrap()
    };
    let mut buff_b : opencl3::memory::Buffer<i32> = unsafe {
        opencl3::memory::Buffer::create(&context, read_flags, 1024, vec_b.as_mut_ptr() as *mut c_void).unwrap()
    };
    let mut buff_c : opencl3::memory::Buffer<i32> = unsafe {
        opencl3::memory::Buffer::create(&context, write_flags, 1024, vec_c.as_mut_ptr() as *mut c_void).unwrap()
    };
    println!("Buffers Created");
    let program = Program::create_and_build_from_source(&context, MY_PROGRAM, "").unwrap();
    println!("Program Created");
    let build_status = program.get_build_status(dev_id).unwrap();
    println!("build_status : {}", build_status);
    let log = program.get_build_log(dev_id).unwrap();
    println!("log : {}", log);
    println!("num kernels : {}", program.get_num_kernels().unwrap());
    let kernel = Kernel::create(&program, "vadd").unwrap();
    let global_size = 1024;
    println!("Kernel Created");
    unsafe {
        kernel.set_arg(0, &buff_a).unwrap();
        kernel.set_arg(1, &buff_b).unwrap();
        kernel.set_arg(2, &buff_c).unwrap();
        kernel.set_arg(3, &global_size).unwrap();
    }
    println!("Arguments Set");
    let work_group_size = 64;
    println!("Work group size : {}", work_group_size);
    let event = unsafe {
        command_queue.enqueue_nd_range_kernel(kernel.into(), 1, core::ptr::null(), &global_size, &work_group_size, &[]).unwrap()
    };
    let event2 = unsafe {
        command_queue.enqueue_read_buffer(&buff_c, opencl3::memory::CL_TRUE, 0, vec_c.as_mut_slice(), &[cl_event::from(event)]).unwrap();
    };
    for i in &vec_c {
        if *i > 16 {
            break;
        }
        println!("{ }", i);
    }   
}
