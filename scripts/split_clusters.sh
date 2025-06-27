## INPUTS
loc_clust=$1

n_models=20
lists=$(tail -n +19 ${loc_clust}/clust-size.xvg | sort -k2 -n | tail -n ${n_models} | awk '{print $1}' | tr -s "\n" " " )

## split clusters.pdb into individual pdbs per frame, representing the cluster
mkdir ${loc_clust}/frames
v=0
while read line; do
  if [[ $line =~ ^TITLE ]]; then
          v=$((v+1))
          outfile="${loc_clust}/frames/model_${v}.pdb"
          continue
  fi
  for d in ${lists}; do
      if [ ${d} == ${v} ] && [ -n "${outfile}" ]; then
          echo "$line" >> "${outfile}"
      fi;
  done

done < ${loc_clust}/clusters.pdb
