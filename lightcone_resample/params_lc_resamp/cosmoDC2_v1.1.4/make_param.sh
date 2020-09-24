#!/bin/bash 

z_ranges=("0_1" "1_2" "2_3")
steps_list=("499 487 475 464 453 442 432 421 411 401 392 382 373 365 355 347 338 331 323 315 307 300 293 286 279 272 266 259 253 247"  "247 241 235 230 224 219 213 208 203 198 194 189 184 180 176 171 167" "167 163 159 155 151 148 144 141 137 134 131 127 124 121")
gltcs_files=("/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/low_z/galaxy_library/\${step}_mod.hdf5" "/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/high_z/galaxy_library/\${step}_mod.hdf5" "/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/high_z/galaxy_library/\${step}_mod.hdf5")
z_types=("low_z" "high_z" "high_z")
healpix_groups=("10066 10067 10068 10069 10070 10071 10072 10073 10074 10193 10194 10195 10196 10197 10198 10199 10200 10201 10202 10321 10322 10323 10324 10325 10326 10327 10328 10329 10444 10445 10446 10447 10448 10449 10450 10451 10452 8786 8787 8788 8789 8790 8791 8792 8793 8794 8913 8914 8915 8916 8917 8918 8919 8920 8921 9042 9043 9044 9045 9046 9047 9048 9049 9050 9169" "9170 9171 9172 9173 9174 9175 9176 9177 9178 9298 9299 9300 9301 9302 9303 9304 9305 9306 9425 9426 9427 9428 9429 9430 9431 9432 9433 9434 9554 9555 9556 9557 9558 9559 9560 9561 9562 9681 9682 9683 9684 9685 9686 9687 9688 9689 9690 9810 9811 9812 9813 9814 9815 9816 9817 9818 9937 9938 9939 9940 9941 9942 9943 9944 9945 9946")
healpix_group_names=("a" "b" "c" "d" "e" "f")
v1=1
v2=1
v3=4
mkdir logs
mkdir logs/old
rm *cosmo*
echo "#!/bin/bash" >"submit_all.sh"
echo "#!/bin/bash" >"run_all_login.sh"
echo "mv logs/*.err logs/old/." >> "submit_all.sh"
echo "mv logs/*.out logs/old/." >> "submit_all.sh"
for i in "${!z_ranges[@]}";do
    echo "z ${i}"
    z_range=${z_ranges[$i]}
    steps=${steps_list[$i]}
    gltcs_file=${gltcs_files[$i]}
    z_type=${z_types[$i]}
    echo ${z_range}
    echo ${steps}
    echo ${gltcs_file}
    echo ${z_type}
    for j in "${!healpix_groups[@]}";do
	healpix_name=${healpix_group_names[j]}
	healpix_group=${healpix_groups[j]}
	echo "healpix group $healpix_name count: `echo $healpix_group$ | wc -w`"
	
	sed "s/#z_range#/${z_range}/g; s/#step_list#/${steps}/g; s@#gltcs_file#@${gltcs_file}@g; s/#z_type#/${z_type}/g; s/#healpix_group#/${healpix_group}/g; s/#v1#/${v1}/g; s/#v2#/${v2}/g; s/#v3#/${v3}/g"<template.param >cosmoDC2_v${v1}.${v2}.${v3}_z_${z_range}_hp:${healpix_name}.param
	sed "s/#z_range#/${z_range}/g; s/#step_list#/${steps}/g; s@#gltcs_file#@${gltcs_file}@g; s/#z_type#/${z_type}/g; s/#healpix_name#/${healpix_name}/g; s/#v1#/${v1}/g; s/#v2#/${v2}/g; s/#v3#/${v3}/g"<template.sh >run_cosmoDC2_v${v1}.${v2}.${v3}_z_${z_range}_hp:${healpix_name}.sh
	chmod +x run_cosmoDC2_v${v1}.${v2}.${v3}_z_${z_range}_hp:${healpix_name}.sh
    echo "qsub -n 1 -t 720 -o params_lc_resamp/cosmoDC2_v${v1}.${v2}.${v3}/logs/${z_range}_${healpix_name}.out -e params_lc_resamp/cosmoDC2_v${v1}.${v2}.${v3}/logs/${z_range}_${healpix_name}.err  --debuglog=params_lc_resamp/cosmoDC2_v${v1}.${v2}.${v3}/logs/${z_range}_${healpix_name}.cobalt  ./lc_resample.py params_lc_resamp/cosmoDC2_v${v1}.${v2}.${v3}/cosmoDC2_v${v1}.${v2}.${v3}_z_${z_range}_hp:${healpix_name}.param" >> "submit_all.sh"
    echo "./lc_resample.py params_lc_resamp/cosmoDC2_v${v1}.${v2}.${v3}/cosmoDC2_v${v1}.${v2}.${v3}_z_${z_range}_hp:${healpix_name}.param" >> "run_all_login.sh"
    done
done
chmod +x "submit_all.sh"
chmod +x "run_all_login.sh"
mv logs/*.err logs/old/.
mv logs/*.out logs/old/.
mv logs/*.cobalt logs/old/.
mv logs/*.colbalt logs/old/.
