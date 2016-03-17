import cl from 'node-opencl';

const FETCH_PER_WORK_ITEM = 16;
const MAX_WORKGROUP_SIZE = 256;

const NULL_NDRANGE = [ ];

const KERNELS_SOURCE_CODE = `
//-----------------------------------------------------------------------------
// Sanity test kernels.
//
__kernel void test(void) {
  printf("  CL device printf: Hello world!\\n");
}

//-----------------------------------------------------------------------------
// Global bandwidth test kernels.
//
#undef FETCH_2
#undef FETCH_8

#define FETCH_2(sum, id, A, jumpBy)      sum += A[id];   id += jumpBy;   sum += A[id];   id += jumpBy;
#define FETCH_4(sum, id, A, jumpBy)      FETCH_2(sum, id, A, jumpBy);   FETCH_2(sum, id, A, jumpBy);
#define FETCH_8(sum, id, A, jumpBy)      FETCH_4(sum, id, A, jumpBy);   FETCH_4(sum, id, A, jumpBy);

#define FETCH_PER_WI ${FETCH_PER_WORK_ITEM}

// Kernels fetching by local_size offset
__kernel void global_bandwidth_v1_local_offset(__global float *A, __global float *B)
{
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float sum = 0;

    FETCH_8(sum, id, A, get_local_size(0));
    FETCH_8(sum, id, A, get_local_size(0));

    B[get_global_id(0)] = sum;
}

__kernel void global_bandwidth_v2_local_offset(__global float2 *A, __global float *B)
{
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float2 sum = 0;

    FETCH_8(sum, id, A, get_local_size(0));
    FETCH_8(sum, id, A, get_local_size(0));

    B[get_global_id(0)] = (sum.S0) + (sum.S1);
}

__kernel void global_bandwidth_v4_local_offset(__global float4 *A, __global float *B)
{
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float4 sum = 0;

    FETCH_8(sum, id, A, get_local_size(0));
    FETCH_8(sum, id, A, get_local_size(0));

    B[get_global_id(0)] = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3);
}

__kernel void global_bandwidth_v8_local_offset(__global float8 *A, __global float *B)
{
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float8 sum = 0;

    FETCH_8(sum, id, A, get_local_size(0));
    FETCH_8(sum, id, A, get_local_size(0));

    B[get_global_id(0)] = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3) + (sum.S4) + (sum.S5) + (sum.S6) + (sum.S7);
}

__kernel void global_bandwidth_v16_local_offset(__global float16 *A, __global float *B)
{
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float16 sum = 0;

    FETCH_8(sum, id, A, get_local_size(0));
    FETCH_8(sum, id, A, get_local_size(0));

    float t = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3) + (sum.S4) + (sum.S5) + (sum.S6) + (sum.S7);
    t += (sum.S8) + (sum.S9) + (sum.SA) + (sum.SB) + (sum.SC) + (sum.SD) + (sum.SE) + (sum.SF);
    B[get_global_id(0)] = t;
}

// Kernels fetching by global_size offset
__kernel void global_bandwidth_v1_global_offset(__global float *A, __global float *B)
{
    int id = get_global_id(0);
    float sum = 0;

    FETCH_8(sum, id, A, get_global_size(0));
    FETCH_8(sum, id, A, get_global_size(0));

    B[get_global_id(0)] = sum;
}

__kernel void global_bandwidth_v2_global_offset(__global float2 *A, __global float *B)
{
    int id = get_global_id(0);
    float2 sum = 0;

    FETCH_8(sum, id, A, get_global_size(0));
    FETCH_8(sum, id, A, get_global_size(0));

    B[get_global_id(0)] = (sum.S0) + (sum.S1);
}

__kernel void global_bandwidth_v4_global_offset(__global float4 *A, __global float *B)
{
    int id = get_global_id(0);
    float4 sum = 0;

    FETCH_8(sum, id, A, get_global_size(0));
    FETCH_8(sum, id, A, get_global_size(0));

    B[get_global_id(0)] = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3);
}

__kernel void global_bandwidth_v8_global_offset(__global float8 *A, __global float *B)
{
    int id = get_global_id(0);
    float8 sum = 0;

    FETCH_8(sum, id, A, get_global_size(0));
    FETCH_8(sum, id, A, get_global_size(0));

    B[get_global_id(0)] = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3) + (sum.S4) + (sum.S5) + (sum.S6) + (sum.S7);
}

__kernel void global_bandwidth_v16_global_offset(__global float16 *A, __global float *B)
{
    int id = get_global_id(0);
    float16 sum = 0;

    FETCH_8(sum, id, A, get_global_size(0));
    FETCH_8(sum, id, A, get_global_size(0));

    float t = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3) + (sum.S4) + (sum.S5) + (sum.S6) + (sum.S7);
    t += (sum.S8) + (sum.S9) + (sum.SA) + (sum.SB) + (sum.SC) + (sum.SD) + (sum.SE) + (sum.SF);
    B[get_global_id(0)] = t;
}

//-----------------------------------------------------------------------------
// Float compute test kernels. 
//
#undef MAD_4
#undef MAD_16
#undef MAD_64

#define MAD_4(x, y)     x = mad(y, x, y);   y = mad(x, y, x);   x = mad(y, x, y);   y = mad(x, y, x);
#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);

__kernel void compute_sp_v1(__global float *ptr, float _A)
{
    float x = _A;
    float y = (float)get_local_id(0);

    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);

    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);

    ptr[get_global_id(0)] = y;
}


__kernel void compute_sp_v2(__global float *ptr, float _A)
{
    float2 x = (float2)(_A, (_A+1));
    float2 y = (float2)get_local_id(0);

    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);

    ptr[get_global_id(0)] = (y.S0) + (y.S1);
}

__kernel void compute_sp_v4(__global float *ptr, float _A)
{
    float4 x = (float4)(_A, (_A+1), (_A+2), (_A+3));
    float4 y = (float4)get_local_id(0);

    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3);
}

__kernel void compute_sp_v8(__global float *ptr, float _A)
{
    float8 x = (float8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    float8 y = (float8)get_local_id(0);

    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);
    MAD_64(x, y);

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}

__kernel void compute_sp_v16(__global float *ptr, float _A)
{
    float16 x = (float16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    float16 y = (float16)get_local_id(0);

    MAD_64(x, y);
    MAD_64(x, y);

    float2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.SAB) + (y.SCD) + (y.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}
`;

let platformDevices = [ ];      // [ CLPlatform ]

class CLPlatform {
  constructor(platformId, devices) {
    this.id = platformId;
    this.devices = devices;
    this.name = cl.getPlatformInfo(platformId, cl.PLATFORM_NAME);
  }
}

class CLDevice {
  constructor(deviceId) {
    this.id = deviceId;
    this.maxAllocBytes =
      ulongToNumber(cl.getDeviceInfo(deviceId, cl.DEVICE_MAX_MEM_ALLOC_SIZE));
    this.maxWorkItemsPerDim =
      cl.getDeviceInfo(deviceId, cl.DEVICE_MAX_WORK_ITEM_SIZES);
    this.maxWorkGroupSize = Math.min(MAX_WORKGROUP_SIZE,
                                     this.maxWorkItemsPerDim[0]);
    this.name = cl.getDeviceInfo(deviceId, cl.DEVICE_NAME);
    this.nrComputeUnits =
      cl.getDeviceInfo(deviceId, cl.DEVICE_MAX_COMPUTE_UNITS);
    switch (cl.getDeviceInfo(deviceId, cl.DEVICE_TYPE)) {
    case cl.DEVICE_TYPE_CPU:
      this.globalBandwidthTestIterations = 20;
      this.computeGroupsPerUnit = 512;
      this.computeTestInterations = 10;
      this.type = 'cpu';
      break;
    case cl.DEVICE_TYPE_GPU:
      this.globalBandwidthTestIterations = 50;
      this.computeGroupsPerUnit = 2048;
      this.computeTestInterations = 30;
      this.type = 'gpu';
      break;
    case cl.DEVICE_TYPE_ACCELERATOR:
      // TODO
      this.type = 'accelerator';
      break;
    default:
      // TODO: do any devices in the field report this type?
      this.type = '(default)';
      break;
    }
    this.transferBandwidthTestIterations = 20;
    this.kernelLatencyTestIterations = 20000;
  }
}

let discoverPlatforms = () => {
  let platforms = cl.getPlatformIDs();
  platforms.forEach((platformId, idx) => {
    let devices = cl.getDeviceIDs(platformId, cl.DEVICE_TYPE_ALL)
          .map((deviceId) => new CLDevice(deviceId));
    let platform = new CLPlatform(platformId, devices);
    platformDevices.push(platform);

    console.log(`====================`);
    console.log(`[${idx}] Platform ${platform.name}`);
    platform.devices.forEach((device, deviceIdx) => {
      console.log(`  [${idx}.${deviceIdx}] ${device.name}`);
    });
    console.log(`====================`);
    console.log(``);
  });
};

let dumpDevice = (platformIdx, deviceIdx) => {
  let platform = platformDevices[platformIdx];
  let device = platform.devices[deviceIdx];
  let id = device.id;
  let exts = cl.getDeviceInfo(id, cl.DEVICE_EXTENSIONS);
  console.log(`  [${platformIdx}.${deviceIdx}] ${device.name}`);
  console.log(`  ----------`);
  console.log(`    CL version\t\t${cl.getDeviceInfo(id, cl.DEVICE_VERSION)}`);
  console.log(`    vendor, id\t\t${cl.getDeviceInfo(id, cl.DEVICE_VENDOR)} 0x${cl.getDeviceInfo(id, cl.DEVICE_VENDOR_ID).toString(16)}`);
  console.log(`    driver\t\t${cl.getDeviceInfo(id, cl.DRIVER_VERSION)}`);
  console.log(`    type\t\t${device.type}`);
  console.log(`    profile\t\t${cl.getDeviceInfo(id, cl.DEVICE_PROFILE)}`);
  console.log(`    float64?\t\t${exts.includes('cl_khr_fp64') ? 'yes' : 'no'}`);
  console.log(`    comp units\t\t${device.nrComputeUnits}`);
  console.log(`    max clock\t\t${cl.getDeviceInfo(id, cl.DEVICE_MAX_CLOCK_FREQUENCY)} MHz`);
  console.log(`    max alloc\t\t${device.maxAllocBytes / Math.pow(2, 30)} GiB`);
  console.log(`    global mem\t\t${ulongToNumber(cl.getDeviceInfo(id, cl.DEVICE_GLOBAL_MEM_SIZE)) / Math.pow(2, 30)} GiB`);
  console.log(`    local mem\t\t${ulongToNumber(cl.getDeviceInfo(id, cl.DEVICE_LOCAL_MEM_SIZE)) / Math.pow(2, 10)} KiB`);
  console.log(`    work group\t\t${device.maxWorkGroupSize}`);
  console.log(`    align bits\t\t${cl.getDeviceInfo(id, cl.DEVICE_MEM_BASE_ADDR_ALIGN)}`);
  let images = cl.getDeviceInfo(id, cl.DEVICE_IMAGE_SUPPORT);
  console.log(`    images?\t\t${images ? 'yes' : 'no'}`);
  if (images) {
    console.log(`      2d max\t\t${cl.getDeviceInfo(id, cl.DEVICE_IMAGE2D_MAX_WIDTH)} x ${cl.getDeviceInfo(id, cl.DEVICE_IMAGE2D_MAX_HEIGHT)}`);
    console.log(`      3d max\t\t${cl.getDeviceInfo(id, cl.DEVICE_IMAGE3D_MAX_WIDTH)} x ${cl.getDeviceInfo(id, cl.DEVICE_IMAGE3D_MAX_HEIGHT)} x ${cl.getDeviceInfo(id, cl.DEVICE_IMAGE3D_MAX_DEPTH)}`);
  }
  console.log(`    vector widths:`);
  {
    console.log(`      char\t\t${cl.getDeviceInfo(id, cl.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR)}`);
    console.log(`      short\t\t${cl.getDeviceInfo(id, cl.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT)}`);
    console.log(`      int\t\t${cl.getDeviceInfo(id, cl.DEVICE_PREFERRED_VECTOR_WIDTH_INT)}`);
    console.log(`      long\t\t${cl.getDeviceInfo(id, cl.DEVICE_PREFERRED_VECTOR_WIDTH_LONG)}`);
    console.log(`      float\t\t${cl.getDeviceInfo(id, cl.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)}`);
    console.log(`      double\t\t${cl.getDeviceInfo(id, cl.DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)}`);
  }
  console.log(`    queue prop:`);
  {
    let prop = cl.getDeviceInfo(id, cl.DEVICE_QUEUE_PROPERTIES);
    console.log(`      OOOrder?\t\t${prop & cl.QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ? 'yes' : 'no'}`);
    console.log(`      prof?\t\t${prop & cl.QUEUE_PROFILING_ENABLE ? 'yes' : 'no'}`);
  }
  console.log(`    timer res\t\t${cl.getDeviceInfo(id, cl.DEVICE_PROFILING_TIMER_RESOLUTION)} ns`);
  console.log(`    const args\t\t${cl.getDeviceInfo(id, cl.DEVICE_MAX_CONSTANT_ARGS)}`);
  console.log(`    samplers\t\t${cl.getDeviceInfo(id, cl.DEVICE_MAX_SAMPLERS)}`);
  console.log(`    exekernel?\t\t${cl.getDeviceInfo(id, cl.DEVICE_EXECUTION_CAPABILITIES) & cl.EXEC_KERNEL ? 'yes' : 'no'}`);
  console.log(`    exenative?\t\t${cl.getDeviceInfo(id, cl.DEVICE_EXECUTION_CAPABILITIES) & cl.EXEC_NATIVE_KERNEL ? 'yes' : 'no'}`);
  console.log(`    endian\t\t${cl.getDeviceInfo(id, cl.DEVICE_ENDIAN_LITTLE) ? 'little' : 'big'}`);
  console.log(`    addr bits\t\t${cl.getDeviceInfo(id, cl.DEVICE_ADDRESS_BITS)}`);
  console.log(`    exts\t\t${exts}`);
  console.log(``);
};

let generateFloats = (nrFloats) => {
  let floats = new Float32Array(nrFloats);  
  for (let i = 0; i < floats.length; ++i) {
    floats[i] = i;
  }
  return floats;
};

let nowNs = () => {
  let timeNs = process.hrtime();
  return timeNs[0] * 1e9 + timeNs[1];
};

let roundToPowerOfTwo = (x) => (
  Math.pow(2, Math.ceil(Math.log(x) / Math.log(2))) | 0
);

let ulongToNumber = (ulong) => {
  return ulong[0] * Math.pow(2, 32) + ulong[1];
};

//-----------------------------------------------------------------------------

let runAndTimeKernelIterations = (Q, kernel, globalWorkSize, localWorkSize, nrIterations) => {
  cl.enqueueNDRangeKernel(Q, kernel, 1, null,
                          [ globalWorkSize ], [ localWorkSize ]);
  cl.enqueueNDRangeKernel(Q, kernel, 1, null,
                          [ globalWorkSize ], [ localWorkSize ]);
  cl.finish(Q);

  let startNs = nowNs();
  for (let i = 0; i < nrIterations; ++i) {
    cl.enqueueNDRangeKernel(Q, kernel, 1, null,
                            [ globalWorkSize ], [ localWorkSize ]);
  }
  cl.finish(Q);
  return (nowNs() - startNs) / nrIterations;
};

let runSanityTest = (device, ctx, program, Q) => {
  console.log('Testing sanity ...');

  let kernel = cl.createKernel(program, 'test');
  let globalWorkSize = 1;
  let localWorkSize = 1;
  let nrIterations = 10;
  let timeNs = runAndTimeKernelIterations(Q, kernel,
                                        globalWorkSize, localWorkSize,
                                        nrIterations);
  console.log(`  ran ${nrIterations} iterations, ${timeNs / 1e6} ms per run`);

  console.log('  ok\n');
};

let runBandwidthSubtestGlobal = (device, ctx, program, Q, vectorWidth, nrItems, inputBuf, outputBuf, localWorkSize, nrIterations) => {
  let localOffsetKernel =
        cl.createKernel(program,
                        `global_bandwidth_v${vectorWidth}_local_offset`);
  cl.setKernelArg(localOffsetKernel, 0, 'float*', inputBuf);
  cl.setKernelArg(localOffsetKernel, 1, 'float*', outputBuf);

  let globalOffsetKernel =
        cl.createKernel(program,
                        `global_bandwidth_v${vectorWidth}_global_offset`);
  cl.setKernelArg(globalOffsetKernel, 0, 'float*', inputBuf);
  cl.setKernelArg(globalOffsetKernel, 1, 'float*', outputBuf);

  let globalWorkSize = nrItems / vectorWidth / FETCH_PER_WORK_ITEM;
  let localOffsetTimeNs =
        runAndTimeKernelIterations(Q, localOffsetKernel,
                                   globalWorkSize, localWorkSize,
                                   nrIterations);
  let globalOffsetTimeNs =
        runAndTimeKernelIterations(Q, globalOffsetKernel,
                                   globalWorkSize, localWorkSize,
                                   nrIterations);
  let timeSec = Math.min(localOffsetTimeNs, globalOffsetTimeNs) / 1e9;
  return nrItems * cl.size_FLOAT / timeSec;
};

let runBandwidthTestGlobal = (device, ctx, program, Q) => {
  console.log('Testing __global memory bandwidth ...');

  let nrIterations = device.globalBandwidthTestIterations;
  let maxItems = device.maxAllocBytes / cl.size_FLOAT / 2;
  let nrItems = maxItems;
  if (device.type === 'cpu') {
    nrItems = Math.min(nrItems, Math.pow(2, 25));
  }
  let itemsSizeBytes = nrItems * cl.size_FLOAT;
  let items = generateFloats(nrItems);

  console.log(`  allocating ${nrItems} floats (${itemsSizeBytes / Math.pow(2, 20)} MiB)`);
  let inputBuf = cl.createBuffer(ctx, cl.MEM_READ_ONLY, itemsSizeBytes);
  let outputBuf = cl.createBuffer(ctx, cl.MEM_WRITE_ONLY, itemsSizeBytes);
  let localWorkSize = device.maxWorkGroupSize;
  let globalWorkSize;

  cl.enqueueWriteBuffer(Q, inputBuf, cl.TRUE, 0, itemsSizeBytes, items);
  [ 1, 2, 4, 8, 16 ].forEach((vectorWidth) => {
    let bytesPerSecond =
          runBandwidthSubtestGlobal(device, ctx, program, Q,
                                    vectorWidth,
                                    nrItems, inputBuf, outputBuf,
                                    localWorkSize, nrIterations);
    console.log(`  vector width ${vectorWidth}: ${(bytesPerSecond / Math.pow(2, 30)).toFixed(3)} GiBps`);
  });

  console.log('  ok\n');
};

let runBandwidthTestTransfer = (device, ctx, program, Q) => {
  console.log('Testing transfer bandwidth ...');

  console.log(`  (TODO)`);

  console.log('  ok\n');
};

let runComputeSubtestFloat = (device, ctx, program, Q, vectorWidth, outputBuf, A, globalWorkSize, localWorkSize, nrIterations) => {
  let kernel = cl.createKernel(program, `compute_sp_v${vectorWidth}`);
  cl.setKernelArg(kernel, 0, 'float*', outputBuf);
  cl.setKernelArg(kernel, 1, 'float', A);

  const workPerWorkItem = 4096;
  let timeNs = runAndTimeKernelIterations(Q, kernel,
                                          globalWorkSize, localWorkSize,
                                          nrIterations);
  let timeSec = timeNs / 1e9;
  return globalWorkSize * workPerWorkItem / timeSec;
};

let runComputeTestFloat = (device, ctx, program, Q) => {
  console.log('Testing compute performance for floats ...');

  const A = 1.3;
  let nrIterations = device.computeTestInterations;
  let globalWorkItems = device.nrComputeUnits * device.computeGroupsPerUnit * device.maxWorkGroupSize;
  let t = Math.min(globalWorkItems * cl.size_FLOAT, device.maxAllocBytes);
  t = roundToPowerOfTwo(globalWorkItems) / cl.size_FLOAT;
  globalWorkItems = t / cl.size_FLOAT;
  let itemsSizeBytes = globalWorkItems * cl.size_FLOAT;

  console.log(`  allocating ${globalWorkItems} floats (${itemsSizeBytes / Math.pow(2, 20)} MiB)`);
  let outputBuf = cl.createBuffer(ctx, cl.MEM_WRITE_ONLY, itemsSizeBytes);
  let globalWorkSize = globalWorkItems;
  let localWorkSize = device.maxWorkGroupSize;

  [ 1, 2, 4, 8, 16 ].forEach((vectorWidth) => {
    let flops = runComputeSubtestFloat(device, ctx, program, Q,
                                       vectorWidth,
                                       outputBuf, A,
                                       globalWorkSize, localWorkSize,
                                       nrIterations);
    console.log(`  vector width ${vectorWidth}: ${(flops / 1e9).toFixed(3)} GFLOPS`);
  });

  console.log('  ok\n');
};

let runComputeTestInt = (device, ctx, program, Q) => {
  console.log('Testing compute performance for ints ...');

  console.log(`  (TODO)`);

  console.log('  ok\n');
};

let runLatencyTestKernel = (device, ctx, program, Q) => {
  console.log('Testing kernel latency ...');

  console.log(`  (TODO)`);

  console.log('  ok\n');
};

let runPerfTests = (platformIdx, deviceIdx) => {
  let platform = platformDevices[platformIdx];
  let device = platform.devices[deviceIdx];

  console.log(`Testing device ${platform.name} / ${device.name} which is a ${device.type}`);
  dumpDevice(platformIdx, deviceIdx);

  let ctx = cl.createContext([ cl.CONTEXT_PLATFORM, platform.id ],
                             [ device.id ]);
  let program = cl.createProgramWithSource(ctx, KERNELS_SOURCE_CODE);
  cl.buildProgram(program);
  let Q;
  if (cl.createCommandQueueWithProperties !== undefined) {
    Q = cl.createCommandQueueWithProperties(ctx, device.id, []); // OpenCL 2
  } else {
    Q = cl.createCommandQueue(ctx, device.id, null); // OpenCL 1.x
  }

  runSanityTest(device, ctx, program, Q);
  runBandwidthTestGlobal(device, ctx, program, Q);
  runBandwidthTestTransfer(device, ctx, program, Q);
  runComputeTestFloat(device, ctx, program, Q);
  runComputeTestInt(device, ctx, program, Q);
  runLatencyTestKernel(device, ctx, program, Q);

  cl.releaseAll();
};

let usage = () => {
  console.error(`Usage: npm test [<platform-id>=0] <device-id>`);
};

let main = (args) => {
  discoverPlatforms();

  args.forEach((arg) => {
    let ids = arg.split(',');
    if (!(1 <= ids.length && ids.length <= 2)) {
      return usage();
    }
    let platform = ids.length === 2 ? parseInt(ids.shift()) : 0;
    let device = parseInt(ids[0]);
    return runPerfTests(platform, device);
  });
};

if (require.main === module) {
  main(process.argv.slice(2));
}
