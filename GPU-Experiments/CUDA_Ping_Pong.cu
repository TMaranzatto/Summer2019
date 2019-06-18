


flipBit = false;

__device__ void dflip(bool* ref){
    ref = !ref;
}

__host__ void hflip(bool* ref){
    ref = !ref;
}

int main(int rounds){
    bool flipBit;
    cudaMallocManaged(&flipBit, sizeof(bool));

    //begin timer
    for(int i = 0; i < rounds; i++){
        dflip(&flipBit);
        hflip(&flipBit);
    }
    //end timer
    //cout >> (time1 - time0);
    return 0;
}