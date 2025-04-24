#mapLayer module writes the subcircuit netlist for each of the layers separately

import random
import numpy as np
import resource
import os

# FIXME: I think they do some fundamentally interesting stuff here...
#        First off, they consider ideal op-amps.
#        Second off, they don't actually have 32x32 partitions, but make their partitions such that they fit the actual guy, which is incorrect.

# Quick Band Aid Fix to Make My Life a little easier :)
# Increase the number of files I can open at a time 
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"Soft limit: {soft}")
print(f"Hard limit: {hard}")

resource.setrlimit(resource.RLIMIT_NOFILE, (min(20000, hard), hard))  # Change 4096 as needed

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"Soft limit: {soft}")
print(f"Hard limit: {hard}")


def find_partition(partitions, index):
    """
    Given a list of split indices and a target index,
    returns (partition_number, index_within_partition).

    # Note Indices are 1-indexed, and partitions are non-inclusive
    # That is, 1:30 means that the partition is from 1 to 29
    
    Example:
    partitions = [1, 30, 50]
    index = 35
    => (2, 6)
    """
    for i in range(len(partitions) - 1):
        start = partitions[i]
        end = partitions[i + 1]
        if start <= index < end:
            return i+1, (index - start) + 1
    raise ValueError(f"Index {index} not found in any partition range.")


def mapPartition(layer1,layer2, xbar_length, LayerNUM,hpar,vpar,metal,T,H,L,W,D,eps,rho,weight_var,data_dir,spice_dir): 
    # updating the resistivity for specific technology node
    l0 = 39e-9 # Mean free path of electrons in Cu
    d = metal # average grain size, equal to wire width
    p=0.25 # specular scattering fraction
    R=0.3 # probability for electron to reflect at the grain boundary
    alpha = l0*R/(d*(1-R)) # parameter for MS model
    dsur_scatt = 0.75*(1-p)*l0/metal # surface scattering
    dgrain_scatt = pow((1-3*alpha/2+3*pow(alpha,2)-3*pow(alpha,3)*np.log(1+1/alpha)),-1) # grain boundary scattering
    rho_new = rho * (dsur_scatt + dgrain_scatt) # new resistivity
    layer1_wb = layer1+1 # number of bitcell in a row including weights and bias

    # Determine where vertical and horizontal partitions are (from IMAC Sim)
    # NOTE: if [1,30, 58], the first partition is 1,29, and the second is 30, 57 :) as the top end is noninclusive.
    horizontal_cuts = [1]

    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number
    posw_r=open(data_dir+'/'+'posweight'+str(LayerNUM)+".txt", "r") # read the positive line conductances
    for line in posw_r:
        if (float(line)!=0):
            if (r < layer2+1):
                r+=1
            else:
                c+=1
                r=1
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    print("positive increase horizontal partition")
                    print(f"row: {r}, col: {c}")
                    n_hpar+=1

                    # Horizontal Partition Here
                    horizontal_cuts.append(c)
                r+=1
        else:
            r+=1
    horizontal_cuts.append(layer1_wb+1)
    posw_r.close()

    # Formally Print horizontal cuts :)
    output = f"Horizontal partitions Layer: {LayerNUM}: "
    output += " | ".join(f"[{horizontal_cuts[i]}:{horizontal_cuts[i+1]-1}]" for i in range(len(horizontal_cuts) - 1))
    print(output)
    print(f"Difference: {np.diff(horizontal_cuts)}")
    print(f"Note: Last Horizontal Partition gets the bias line")

    # Determine Vertical Partitions :)

    vertical_cuts = [1]
    
    # writing the circuit for vertical line parasitic resistances
    parasitic_res = rho_new*W/(metal*T)
    for i in range(layer1_wb):
        n_vpar=1 # vertical partition number
        c=i+1 # column number
        for j in range(layer2):
            r=j+1 # row number
            if (i == layer1): # only for the bias line (last line to write btw)
                if (j == 0):
                    temp=1
                elif (j == int(layer2*n_vpar/vpar+min((layer2%vpar)/n_vpar,1))):
                    temp=1
                    n_vpar+=1
                    vertical_cuts.append(r)
                else:
                    temp=1
    
    vertical_cuts.append(layer2+1)
    output = f"Vertical partitions Layer: {LayerNUM}: "
    output += " | ".join(f"[{vertical_cuts[i]}:{vertical_cuts[i+1]-1}]" for i in range(len(vertical_cuts) - 1))
    print(output)
    print(f"Difference: {np.diff(vertical_cuts)}")
    print(f"Note: Each Vertical Guy gets own bias. Each Partition's initial wire connects to vdd")

    # Open file descriptors for all possible things that we need to open (Each guy is xbar_length x 1) :)
    # File descriptor named: partioned_layer_{layer_num}_{hpar}_{vpar}_{split_vpar}.sp
    # We split vpar since we have 32x1 based mac units.
    open_fd = {}    # Index into this using hpar, vpar, split_vpar index

    for x_id in range(hpar):
        for y_id in range(vpar):
            # Vertical refers to having to split up input into multiple :)
            # In our case, even though we will have the xbar_length x xbar_length, we will split to 32x1 for parallezability
            # Get vertical cuts length :)
            new_range_low = vertical_cuts[y_id]-1
            new_range_high = vertical_cuts[y_id+1]-1


            for split_vpar in range(new_range_low, new_range_high):
                file_template = f"partitioned_layer_{LayerNUM}_{x_id+1}_{y_id+1}_{split_vpar+1}.sp"
                fd = open(os.path.join(spice_dir,file_template), "w")

                # Write subcircuit definition :)
                fd.write(f".SUBCKT layer{LayerNUM}_{x_id+1}_{y_id+1}_{split_vpar+1}"+" vdd vss 0 ")

                # Number of input and output in the circuit definition :)
                for i in range(xbar_length):
                    fd.write("in%d "%(i+1))
                
                fd.write("out%d"%(1))

                fd.write("\n\n**********Positive Weighted Array**********\n")

                open_fd[(x_id+1, y_id+1, split_vpar+1)] = fd    

    
    # Open all the weights and bias files :)
    posw_r=open(data_dir+'/'+'posweight'+str(LayerNUM)+".txt", "r") # read the positive line conductances
    negw_r=open(data_dir+'/'+'negweight'+str(LayerNUM)+".txt", "r")
    posb_r=open(data_dir+'/'+'posbias'+str(LayerNUM)+".txt", "r")
    negb_r=open(data_dir+'/'+'negbias'+str(LayerNUM)+".txt", "r")


    # Go through and write each of the files :)
    # TODO:


    # Write Positive Array 
    # FIXME: Last horizontal partition still needs the bias :)
    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number
    for line in posw_r:
        if (float(line)!=0):
            if (r < layer2+1):
                # Calculate Indices :) (Row is vertical partitioning, while column is vertical partitioning)
                y_id, split_r = find_partition(vertical_cuts, r)
                x_id, split_c = find_partition(horizontal_cuts, c)
                #print(f"Row: {r}, Col: {c}, X_ID: {x_id}, Y_ID: {y_id}, X_PAR: {split_c}, Y_PAR: {split_r}")
                open_fd[(x_id, y_id, split_r)].write("Rwpos%d_%d in%d_%d sp%d_%d %f\n"% (split_c,split_r, split_c,split_r,split_c,split_r,float(line)))

                #layer_w.write("Rwpos%d_%d_%d in%d_%d sp%d_%d %f\n"% (c,r, n_hpar, c,r,c,r,float(line)))
                r+=1;
            else:
                c+=1;
                r=1;
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    print("positive increase horizontal partition")
                    print(f"row: {r}, col: {c}")
                    n_hpar+=1

                # Calculate Indices :) (Row is vertical partitioning, while column is vertical partitioning)
                y_id, split_r = find_partition(vertical_cuts, r)
                x_id, split_c = find_partition(horizontal_cuts, c)

                #print(f"Row: {r}, Col: {c}, X_ID: {x_id}, Y_ID: {y_id}, X_PAR: {split_c}, Y_PAR: {split_r}")

                open_fd[(x_id, y_id, split_r)].write("Rwpos%d_%d in%d_%d sp%d_%d %f\n"% (split_c,split_r, split_c,split_r,split_c,split_r,float(line)))
                #layer_w.write("Rwpos%d_%d_%d in%d_%d sp%d_%d %f\n"% (c,r,n_hpar, c,r,c,r,float(line)))
                r+=1;
        else:
            r+=1;

    # Write Negative Array
    for key, file in open_fd.items():
        try:
            file.write("\n\n**********Negative Weighted Array**********\n")

        except Exception as e:
            print(f"Error {key}: {e}")
            
    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number
    for line in negw_r:
        if (float(line)!=0):
            if (r < layer2+1):
                y_id, split_r = find_partition(vertical_cuts, r)
                x_id, split_c = find_partition(horizontal_cuts, c)
                #print(f"Row: {r}, Col: {c}, X_ID: {x_id}, Y_ID: {y_id}, X_PAR: {split_c}, Y_PAR: {split_r}")
                open_fd[(x_id, y_id, split_r)].write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (split_c,split_r,split_c,split_r,split_c,split_r,float(line)))

                #layer_w.write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (c,r,c,r,c,r,float(line)))
                r+=1;
            else:
                c+=1;
                r=1;
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    n_hpar+=1

                y_id, split_r = find_partition(vertical_cuts, r)
                x_id, split_c = find_partition(horizontal_cuts, c)
                #print(f"Row: {r}, Col: {c}, X_ID: {x_id}, Y_ID: {y_id}, X_PAR: {split_c}, Y_PAR: {split_r}")
                open_fd[(x_id, y_id, split_r)].write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (split_c,split_r,split_c,split_r,split_c,split_r,float(line)))


                #layer_w.write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (c,r,c,r,c,r,float(line)))
                r+=1;
        else:
            r+=1;	
    

    # Write Bias (if applicable)
    # writing the circuit for positive line biases
    # This only applies for the last layer :)
    # TODO: Start off here on the biases :)
    layer_w.write("\n\n**********Positive Biases**********\n\n")
    r=1
    for line in posb_r:
        if (float(line)!=0):
            layer_w.write("Rbpos%d vd%d sp%d_%d %f\n"% (r,r,horizontal_cuts[-1],r,float(line)))
            r+=1
        else:
            r+=1


    # Write 






    # Close All File Descriptors :)
    posw_r.close()
    negw_r.close()
    posb_r.close()
    negb_r.close()


    # TODO: Close all other file descriptors
    # Close all file descriptors
    for key, file in open_fd.items():
        try:
            file.close()
            print(f"Closed: {key}")
        except Exception as e:
            print(f"Error closing {key}: {e}")

            



















    exit()






                









    # writing the circuit for positive line weights
    layer_w.write("\n\n**********Positive Weighted Array**********\n")
    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number
    for line in posw_r:
        if (float(line)!=0):
            if (r < layer2+1):
                layer_w.write("Rwpos%d_%d_%d in%d_%d sp%d_%d %f\n"% (c,r, n_hpar, c,r,c,r,float(line)))
                r+=1;
            else:
                c+=1;
                r=1;
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    print("positive increase horizontal partition")
                    print(f"row: {r}, col: {c}")
                    n_hpar+=1
                layer_w.write("Rwpos%d_%d_%d in%d_%d sp%d_%d %f\n"% (c,r,n_hpar, c,r,c,r,float(line)))
                r+=1;
        else:
            r+=1;
    
    
    # writing the circuit for negative line weights
    layer_w.write("\n\n**********Negative Weighted Array**********\n\n")
    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number
    for line in negw_r:
        if (float(line)!=0):
            if (r < layer2+1):
                layer_w.write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (c,r,c,r,c,r,float(line)))
                r+=1;
            else:
                c+=1;
                r=1;
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    n_hpar+=1
                layer_w.write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (c,r,c,r,c,r,float(line)))
                r+=1;
        else:
            r+=1;	
    
    
    # writing the circuit for positive line biases
    layer_w.write("\n\n**********Positive Biases**********\n\n")
    r=1
    for line in posb_r:
        if (float(line)!=0):
            layer_w.write("Rbpos%d vd%d sp%d_%d %f\n"% (r,r,layer1_wb,r,float(line)))
            r+=1
        else:
            r+=1
    
    
    # writing the circuit for negative line biases
    layer_w.write("\n\n**********Negative Biases**********\n\n")
    r=1
    for line in negb_r:
        if (float(line)!=0):
            layer_w.write("Rbneg%d vd%d sn%d_%d %f\n"% (r,r,layer1_wb,r,float(line)))
            r+=1
        else:
            r+=1
    























    
    # writing the circuit for vertical line parasitic resistances
    layer_w.write("\n\n**********Parasitic Resistances for Vertical Lines**********\n\n")
    parasitic_res = rho_new*W/(metal*T)
    for i in range(layer1_wb):
        n_vpar=1 # vertical partition number
        c=i+1 # column number
        for j in range(layer2):
            r=j+1 # row number
            if (i == layer1): # only for the bias line
                if (j == 0):
                    layer_w.write("Rbias%d vdd vd%d %f\n"% (r,r,parasitic_res))
                elif (j == int(layer2*n_vpar/vpar+min((layer2%vpar)/n_vpar,1))):
                    layer_w.write("Rbias%d vdd vd%d %f\n"% (r,r,parasitic_res))
                    n_vpar+=1
                else:
                    layer_w.write("Rbias%d vd%d vd%d %f\n"% (r,j,r,parasitic_res))
            
            else: # the input connected vertical lines
                if (j == 0):
                    layer_w.write("Rin%d_%d in%d in%d_%d %f\n"% (c,r,c,c,r,parasitic_res))
                elif (j == int(layer2*n_vpar/vpar+min((layer2%vpar)/n_vpar,1))):
                    layer_w.write("Rin%d_%d in%d in%d_%d %f\n"% (c,r,c,c,r,parasitic_res))
                    n_vpar+=1
                else:
                    layer_w.write("Rin%d_%d in%d_%d in%d_%d %f\n"% (c,r,c,j,c,r,parasitic_res))
    
    # writing the circuit for horizontal line parasitic resistances
    layer_w.write("\n\n**********Parasitic Resistances for I+ and I- Lines****************\n\n")
    parasitic_res = rho_new*L/(metal*T)
    n_hpar=1 # horizontal partition number
    for i in range(layer1_wb):
        c=i+1 # column number
        for j in range(layer2):
            r=j+1 # row number
            if (i == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)-1)):
                if (i == layer1):
                    layer_w.write("Rsp%d_%d sp%d_%d sp%d_p%d %f\n"% (c,r,c,r,r,n_hpar,parasitic_res))
                    layer_w.write("Rsn%d_%d sn%d_%d sn%d_p%d %f\n"% (c,r,c,r,r,n_hpar,parasitic_res))
                else:
                    layer_w.write("Rsp%d_%d sp%d_%d sp%d_p%d %f\n"% (c,r,c,r,r,n_hpar,parasitic_res))
                    layer_w.write("Rsn%d_%d sn%d_%d sn%d_p%d %f\n"% (c,r,c,r,r,n_hpar,parasitic_res))
                    if (j == layer2-1):
                        n_hpar+=1;
            else:
                layer_w.write("Rsp%d_%d sp%d_%d sp%d_%d %f\n"% (c,r,c,r,c+1,r,parasitic_res))
                layer_w.write("Rsn%d_%d sn%d_%d sn%d_%d %f\n"% (c,r,c,r,c+1,r,parasitic_res))





    # writing the circuit for Op-AMPS and connecting resistors
    layer_w.write("\n\n**********Weight Differntial Op-AMPS and Connecting Resistors****************\n\n")
    for i in range(hpar):
        for j in range(layer2):
            layer_w.write("XDIFFw%d_p%d sp%d_p%d sn%d_p%d nin%d_%d diff%d\n"% (j+1,i+1,j+1,i+1,j+1,i+1,j+1,i+1,LayerNUM))
            layer_w.write("Rconn%d_p%d nin%d_%d nin%d 1m\n"% (j+1,i+1,j+1,i+1,j+1))
    
    
    
    # writing the circuit for neurons
    layer_w.write("\n\n**********neurons****************\n\n")	
    for i in range(layer2):
        layer_w.write("Xsig%d nin%d out%d vdd 0 neuron\n"% (i+1,i+1,i+1))
    
    
    layer_w.write(".ENDS layer"+ str(LayerNUM))
    layer_w.close()

# Test Function
mapPartition(84, 10, 32, 3, 3, 1, 2.69999999997e-8, 2.2e-8, 2e-8, 1.35e-7, 1.08e-7, 4.5e-8, 1.77079999999e-10, 1.9e-8, 0, 'data', 'test_spice')
#Horizontal is 3, vertical is 1