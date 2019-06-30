#include cstdef;

//returns value for threadId%64 if not taken
//else returns NULL
//assumes array is for integers
__global__
void get(int* array, long* bitmap){
    int flag = 0;
    //checking edge case that the array is empty
    //send a message to host
    if (*bitmap == -1){
        //do something
        flag = -1;
    }

    long i = (blockIdx.x*blockDim.x + threadIdx.x)%64;
    long addr = 1<<i;
    atomicOr(*bitmap, addr);
    if((*bitmap>>i)&1 == 0){
        //do something with the value
        flag = 1;
    }
    //else do nothing
}

int main(void){
    long bitmap;
    int arr[64];

    cudaMallocManaged(&bitmap, sizeof(long));
    cudaMallocManaged(&arr, sizeof(arr));

    get<<<1, 256>>>(arr, bitmap);


}