# RTMScore

RTMScore is a a novel scoring function based on residue-atom distance likelihood potential and graph transformer, for the prediction of protein-ligand interactions. 
<div align=center>
<img src="https://github.com/andsanv/RTMScore/blob/main/121.jpg" width="600px" height="300px">
</div> 

The proteins and ligands were first characterized as 3D residue graphs and 2D molecular graphs, respectively, followed by two groups of independent graph transformer layers to learn the node representations of proteins and ligands. Then all node features were concatenated in a pairwise manner, and input into an MDN to calculate the parameters needed for a mixture density model. Through this model, the probability distribution of the minimum distance between each residue and each ligand atom could be obtained, and aggregated into a statistical potential by summing all independent negative log-likelihood values.

### Installation (Apple Silicon)

Clone the repo and cd into it
```
git clone https://github.com/andsanv/RTMScore.git && cd RTMScore
```

Setup the python environment, we used [uv](https://docs.astral.sh/uv/) but you can simply use pip if you prefer
```
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Install openbabel with brew
```
brew install open-babel swig
```

Then, download openbabel from https://pypi.org/project/openbabel/#files, in our case we run the following, simply paste the download link in the placeholder
```
curl [openbabel-3.1.1.1.tar.gz link] --output openbabel-3.1.1.1.tar.gz
tar -xzvf openbabel-3.1.1.1.tar.gz && rm -rf openbabel-3.1.1.1.tar.gz && cd openbabel-3.1.1.1
python setup.py build_ext -I/opt/homebrew/include/openbabel3 -L/opt/homebrew/lib && python setup.py install && cd ..
```

### Datasets
[PDBbind](http://www.pdbbind.org.cn)    
[CASF-2016](http://www.pdbbind.org.cn)    
[PDBbind-CrossDocked-Core](https://zenodo.org/record/5525936)      
[DEKOIS2.0](https://zenodo.org/record/6623202)       
[DUD-E](https://zenodo.org/record/6623202)

### Examples for using the trained model for prediction
```
cd example
```
___# input is protein (need to extract the pocket first)___
```
python rtmscore.py -p ./1qkt_p.pdb -l ./1qkt_decoys.sdf -rl ./1qkt_l.sdf -gen_pocket -c 10.0 -m ../trained_models/rtmscore_model1.pth
```
___# input is pocket___
```
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -m ../trained_models/rtmscore_model1.pth
```
___# calculate the atom contributions of the score___
```
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -ac -m ../trained_models/rtmscore_model1.pth
```
___# calculate the residue contributions of the score___
```
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -rc -m ../trained_models/rtmscore_model1.pth
```