# Policy Gradient


## Install

First, install gym and mujoco environments. You may need to install other dependencies depending on your system.

```
pip install gym
```


and then install Mujoco:
Go to the license page of their website,
https://www.roboti.us/license.html
and read the "MuJoCo Pro Trial License: 30 days" section. Follow the instructions.

You will have to download an executable for your platform,
i.e. if you are using a mac, click the "OSX" button,
and when it has finished downloading, in terminal type

$ cd ~/Downloads
$ chmod u+x getid_osx
$ ./getid_osx

and copy paste the ID it gives you. Enter this into the form on the webpage,
and you will be emailed a file containing a mjkey.txt attachment. Download 
this file.


Now, go to the downloads page of the website and download the mujoco131 file for 
your system. Make sure to download the 131 version.
Make a directory ~/.mujoco, and 
Unzip the downloaded file into ~/.mujoco/mjpro131
Place the license key mjkey.txt at ~/.mujoco/mjkey.txt
```
pip install mujoco-py==0.5.7
```

We also require you to use Tensorflow version>=1.4.


## Environment

### Cartpole-v0
Discrete
### HalfCheetah-v1
Continuous
### InvertedPendulum-v1
Continuous

## Training

Once done with implementing `pg.py`, launch `python pg.py` that will run your code 




**Credits**
Assignment code written by Luke Johnston and Shuhui Qu.