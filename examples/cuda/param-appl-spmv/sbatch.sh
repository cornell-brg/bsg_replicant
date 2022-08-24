#!/bin/bash
#SBATCH --job-name=jOBnAme
#SBATCH --ntasks=1
#SBATCH --mem=4g
#SBATCH --time=168:00:00
#SBATCH --output=jOBnAme.log
pwd; hostname; date

module load scl-8
module load synopsys-2020/synopsys-vcs-R-2020.12
source setup-python3.sh
(cd jOBnAme; pwd; make $1;)

date
