# SNN Samples
## Data description
There are two data files per sample:
* **x_adc**: Data sampled by an traditional ADC
    * \["adc"\]: ushort[,]
* **x_threshold**: Threshold based sampled (Send-on-Delta sampling) with a threshold of 50mV
    * \["spikes"\]: short[,]
* **ecg_x**: Threshold based sampled with 2 leads
    * \["signal"\]: double[,]
    * \["tm"\]: double[,]
* **ecg_x_ann**: Anotations of the R waves
    * \["ann"\]: int[,]
    * \["anntype"\]: string
    * \["subtype"\]: Byte[,]

Files with a matching index where created at the same time by two controllers. 

**Reading**: To read a file, open it with a library which is able to parse Matlab datafiles and read the `adc` or `spike` column (depending on the sample type).
