# Paraview on Polaris

!!! note  
	At this time, we only support client/server mode where the user must manually launch the server on Polaris.


## Setting up Paraview 

From your local client select Connect, either from the File menu, or by clicking on the icon circled below:

![Connect icon](connect-icon.png) 

A new window will open where you can configure a server. Click on Add Server:

![Choose server](choose-server.png)

Give your server a name, select Client/Server, localhost, and a TCP port (8000 in this example)

![Edit server](edit-server.png)

Click "Configure". In the next window there is an option to set up how Paraview server will be launched, and the default is "Manual". Leave it on "Manual" and click "Save".

You will use these settings when establishing the connection.

## Launching the Paraview server on Polaris

You can launch an interactive session on Polaris compute nodes with the following command (adjust parameters as needed to match your allocation, desired number of nodes, queue, walltime, and filesystems):

```shell
qsub -l walltime=01:00:00 -l select=2 -A yourallocation -q debug -I -l filesystems=home:grand
```

When the job starts you will receive a prompt on your head node like this:

```
username@x3005c0s7b0n0:~>
```

Make a note of the node hostname (`x3005c0s7b0n0` in the example above). You can also get this information from `qstat -fx jobID`


Now load the Paraview module

```
username@x3005c0s7b0n0:~> module load paraview

Lmod is automatically replacing "nvhpc/21.9" with "gcc/11.2.0".

----------------------------------------------------------------------------------
     Paraview v5.11.0 successfully loaded
----------------------------------------------------------------------------------

Due to MODULEPATH changes, the following have been reloaded:
  1) cray-mpich/8.1.16
```

and launch the Paraview server with

```
srizzi@x3005c0s7b0n0:~> mpirun -n 8 pvserver --server-port=8000
Waiting for client...
Connection URL: cs://x3005c0s7b0n0:8000
Accepting connection(s): x3005c0s7b0n0:8000
```

In this case `pvserver` will be listening on TCP port 8000 of your head node. You can change this port if you want.

## Creating a tunnel over ssh

We need to establish an ssh tunnel to connect client to server. On your local machine open a new terminal and type:

```
ssh -v -N -L 8000:x3005c0s7b0n0:8000 polaris.alcf.anl.gov
```

where 8000 is a TCP port and `x3005c0s7b0n0` the name of your head node. Adjust these values accordingly.

Among multiple lines with debug information,  you should see something like:

```
debug1: Local connections to LOCALHOST:8000 forwarded to remote address x3005c0s7b0n0:8000
```

Keep this terminal open for the duration of your session to keep the ssh tunnel active.

Now you are ready to launch your Paraview client locally. Keep in mind that client and servers versions must match. The Paraview version currently deployed on Polaris is 5.11.0

## Connecting to Paraview server

Connect your Paraview client to the server configuration you created above. You can select Connect, either from the File menu, or the icon circled in the figure:

![Connect icon](connect-icon.png)  

and selecting the configuration you created in a previous step.

The connection should point to:

```
localhost:8000
```

In the terminal where you launched the server you will see now that the connection is established. Note that Paraview may take a few seconds to connect. This is normal behavior.

```
username@x3005c0s7b0n0:~> mpirun -n 8 pvserver --server-port=8000
Waiting for client...
Connection URL: cs://x3005c0s7b0n0:8000
Accepting connection(s): x3005c0s7b0n0:8000
Client connected.
```

At this point you can use Paraview normally.






