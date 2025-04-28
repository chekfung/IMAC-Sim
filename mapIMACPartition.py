#mapIMAC module connects the hidden layers and sets the configurations in the SPICE netlist
import os
import random
import mapLayer
from mapPartitionIMAC import *


def mapIMAC(nodes,xbar_length,hpar,vpar,metal,T,H,L,W,D,eps,rho,weight_var,testnum,data_dir,spice_dir,vdd,vss,tsampling):
    f=open(spice_dir+'/'+'classifier.sp', "w")

    # Write Header
    f.write("*Fully-connected Classifier\n")
    f.write(".lib './models' ptm14hp\n")    #the transistor library can be changed here (The current format does not use transistor for the weighted array)
    for x in range(len(nodes)-1):		
        f.write('.include diff'+str(x+1)+'.sp\n')
    f.write(".include 'neuron.sp'\n")
    #f.write(".option post \n") # Post causes everything to get spit out :)
    f.write('.option ingold=2 artist=2 psf=2\n')
    f.write('.OPTION DELMAX=1NS\n')
    f.write('.option probe\n')
    f.write('.probe v(output0) \n')
    f.write(".op\n")
    f.write(".PARAM VddVal=%f\n"%vdd)
    f.write(".PARAM VssVal=%f\n"%vss)
    f.write(".PARAM tsampling=%fn\n"%tsampling)

    layers = {}
    layer_cuts = {}

    # Map all the layers
    for i in range(len(nodes)-1):
        horizontal_cuts, vertical_cuts, keys = mapPartition(nodes[i],nodes[i+1],xbar_length, i+1, hpar[i],vpar[i],metal,T,H,L,W,D,eps,rho,weight_var,data_dir,spice_dir)

        layers[i+1] = keys
        layer_cuts[i+1] = (horizontal_cuts, vertical_cuts)

    # Include all Layers to Include :)
    for i in range(len(nodes)-1):
        keys = layers[i+1]

        for x_id, y_id, split_r in keys:
            format_string = f".include layer_{i+1}_{x_id}_{y_id}_{split_r}.sp\n"
            f.write(format_string)

    # Write all invocations of each of the partitioned cross bars :)
    for i in range(len(nodes)-1):
        layer_keys = layers[i+1]
        hor_cut, vert_cut = layer_cuts[i+1]

        # This information technically not needed :)
        layer_num_neurons = nodes[i]
        layer_output_neurons = nodes[i+1]

        # Note: If same layer, same y_id, and same row, go to the same output number :)

        input_name = "layer_{}_in{} "

        if i!=0:
            input_name = "layer_{}_neuron_output_{} "

        print(f"writing layer {i+1}")
        f.write(f"\n\n********** Layer {i+1} **********\n")

        for (x_id, y_id, vpar) in layer_keys:
            # TODO: Change to use separate vdd, vss :)
            f.write(f"Xlayer_{i+1}_{x_id}_{y_id}_{vpar} vdd vss 0 ")

            # Calculate input ranges :)
            low_range_x = hor_cut[x_id-1]
            high_range_x = hor_cut[x_id]

            if x_id == len(hor_cut)-1:
                high_range_x -= 1

            # Calculate y
            low_range_y = vert_cut[y_id-1]
            global_y_value = (low_range_y - 1) + vpar
            
            #print(f"X: {low_range_x} - {high_range_x-1}")
            #print(f"Y: {global_y_value}")

            # Write inputs 
            for j in range(low_range_x, high_range_x):
                f.write(input_name.format(i, j))

            # Write zeroes for 32 crossbar when partitioned funny
            low_index = hor_cut[x_id] - hor_cut[x_id-1]+1

            # On last horizontal index, get rid of bias counting as one of the 32 inputs that we need to write.
            if x_id == len(hor_cut)-1:
                low_index -= 1

            high_index = xbar_length

            #print(low_index, high_index+1)

            for j in range(low_index, high_index+1):
                f.write("0 ")

            # Write output
            f.write(f"layer_{i+1}_neuron_input_{global_y_value} ")
            
            f.write(f"layer_{i+1}_{x_id}_{y_id}_{vpar}\n")

        # Write output neurons :)
        f.write(f"\n********** layer {i+1} neurons****************\n\n")

        for j in range(layer_output_neurons):
            f.write(f"Xsig_layer_{i+1}_{j+1} layer_{i+1}_neuron_input_{j+1} layer_{i+1}_neuron_output_{j+1} vdd 0 neuron\n")


    f.write("\n\n**********Input Test****************\n\n")
    c=open(data_dir+'/'+'data_sim.txt', "r")
    input_str = c.readlines()[0].split()
    input_num = [float(num) for num in input_str]
    for line in range(nodes[0]):
        f.write("v%d layer_0_in%d 0 PWL( 0n 0 "%(line+1,line+1))
        for image in range(testnum):
            f.write("%fn %f %fn %f "%(image*tsampling+0.1,input_num[line+image*nodes[0]],(image+1)*tsampling,input_num[line+image*nodes[0]]))
        f.write(")\n")
    c.close()

	
    f.write("\n\n\nvss vss 0 DC VssVal\n")
    f.write("\n\n\nvdd vdd 0 DC VddVal\n")
    f.write(".TRAN 0.1n %d*tsampling\n"%(testnum))

    # Change to measure energy 
    for i in range(testnum):
        #f.write(".MEASURE TRAN pwr%d AVG POWER FROM=%d*tsampling+0.1n TO=%d*tsampling\n"%(i,i,i+1))
        
        #f.write(".MEASURE TRAN energy%d INTEG POWER FROM=%d*tsampling+0.1n TO=%d*tsampling\n"%(i,i,i+1))
        f.write(".MEASURE TRAN total_energy%d INTEG 'abs(V(vdd)*I(vdd)) + abs(V(vss)*I(vss))' FROM=%d*tsampling TO=%d*tsampling\n"%(i,i,i+1))


    for i in range(testnum):
        for j in range(nodes[len(nodes)-1]):
            f.write(".MEAS TRAN VOUT%d_%d FIND v(output%d) AT=%d*tsampling\n"%(j,i,j,i+1))
    f.write(".end")
    f.close() 
			
			
