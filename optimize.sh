#!/bin/csh
#rm -rf tmp
mkdir -p tmp/run
mkdir -p tmp/optimization

set MODEL_SRC=ESM4_longamip_D1_am4p2_proto7b_isop_whiteCapsAlbedo_salt_ch4_2d_adj_vmr_ceds_v2021_revised_v6c_SIS2_ESM4.2_LM4p1_CH4_track
set ITER_MAX=10

foreach opt (all_src_dep)

set optimization_f=config/optimization/${opt}

set scale_tau_dep_G0=8
set scale_tau_dep_G30=5

foreach geo_source (0 30)
foreach experiment (run_tag_dep_transport.json)
rm -f tmp/run/G${geo_source}_$experiment
cp config/run/$experiment tmp/run/G${geo_source}_$experiment
sed -i "s|MODEL_SRC|${MODEL_SRC}|" tmp/run/G${geo_source}_$experiment
sed -i "s|GEO_SOURCE|${geo_source}|" tmp/run/G${geo_source}_$experiment
if ($geo_source == 0) then
sed -i "s|SCALE_TAU_DEP|${scale_tau_dep_G0}|g" tmp/run/G${geo_source}_$experiment
else if ($geo_source == 5) then
sed -i "s|SCALE_TAU_DEP|${scale_tau_dep_G5}|g" tmp/run/G${geo_source}_$experiment
else if ($geo_source == 10) then
sed -i "s|SCALE_TAU_DEP|${scale_tau_dep_G10}|g" tmp/run/G${geo_source}_$experiment
else if ($geo_source == 20) then
sed -i "s|SCALE_TAU_DEP|${scale_tau_dep_G20}|g" tmp/run/G${geo_source}_$experiment
else if ($geo_source == 30) then
sed -i "s|SCALE_TAU_DEP|${scale_tau_dep_G30}|g" tmp/run/G${geo_source}_$experiment
else
echo "error"
exit
endif
set count=0

#20% std based on Ehhalt all sources
#sqrt(4*4+6*6+3*3+2*2+8*8)/(76-23)

foreach error_src (0.1 0.15 0.2 0.25  0.3)
foreach error_deposition (0.15 0.225 0.3 0.375 0.45)
foreach error_ch4 (0.01 0.025 0.05 0.075 0.1)
foreach gamma_opt (1)


#foreach optimization_f (config/optimization/Anthro_BB_time_Nat_Deposition.json)
set optimization=`basename $optimization_f`
echo $optimization
set opt_file=tmp/optimization/${optimization}_${count}.json
rm -f $opt_file
cp config/optimization/${optimization}.json  $opt_file
#THIS ADDS TWO ERRORS:
#John's error + a relative error (taken to be 2%)
sed -i 's|"OBS_ERROR"|"input/Observations/firn_reconstruction_errors_20231228.csv",\n"error_rel":0.02|' $opt_file
sed -i "s|OBS_SRC|input/Observations/firn_reconstructions_20231017.csv|" $opt_file 
sed -i "s|MODEL_SRC|${MODEL_SRC}|" $opt_file
sed -i 's,"iter_max":1,"iter_max":'"${ITER_MAX}"',' $opt_file
sed -i 's,"it_start":0,"it_start":840,' $opt_file
sed -i 's,"ntimes":1788,"ntimes":876,' $opt_file
sed -i "s|_count|_${count}|" $opt_file
sed -i "s|error_deposition|${error_deposition}|" $opt_file
sed -i "s|error_src|${error_src}|" $opt_file
sed -i "s|error_ch4|${error_ch4}|" $opt_file
sed -i "s|gamma_opt|${gamma_opt}|" $opt_file

#COMMENT OUT TO RUN ON CLUSTER (SLURM)
#sbatch optimize_slurm tmp/run/G${geo_source}_$experiment $opt_file
./optimize_slurm tmp/run/G${geo_source}_$experiment $opt_file

@ count++
end
end
end
end
end
end
end
