# anaconda
anaconda-navigator ----- (open anaconda GUI) <br>


# Linux
sudo -i -------------------------------------- (Enter into admin mode) <br>
sudo lsof -i -P -n | grep LISTEN ------------- (list all ports) <br>
sudo ss --kill state listening src :8888 ----- (close given port) <br>

  
# jupyterlab
jupyter labextension list   						#> (list all jupyterlab extensions) <br>
jupyter lab build   							#> (build jupyter lab)(do after every extension insertion or deletion) <br>
pip install --upgrade jupyterlab   					#> (upgrade jupyter lab) <br>
jupyter labextension uninstall <@jupyter-widgets/jupyterlab-manager>    #> (remove extension) <br>
jupyter server extension enable --py jupyterlab_git   			#> (enable extension) <br>
pip install --upgrade jupyterlab-git   					#> (upgrade extension) <br>


# pip
sudo apt remove <pkg>   						#> (remove package) <br>
pip install --upgrade <pkg>   						#> (upgrade package) <br>


# snap
sudo snap remove <pkg>   						#> (remove package) <br>


# Pendrive
* steps   								#> (format pendrive using NTFS mode) <br>
	1) sudo umount /dev/sdb1 <br>
	2) sudo mkfs.ntfs /dev/sdb1 <br>