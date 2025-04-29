#mapIMAC module connects the hidden layers and sets the configurations in the SPICE netlist

import random
import mapLayer
def mapIMAC(nodes,length,hpar,vpar,metal,T,H,L,W,D,eps,rho,weight_var,testnum,data_dir,spice_dir,vdd,vss,tsampling):
    f=open(spice_dir+'/'+'classifier.sp', "w")
    f.write("*Fully-connected Classifier\n")
    f.write(".lib './models' ptm14hp\n")    #the transistor library can be changed here (The current format does not use transistor for the weighted array)
    for x in range(len(nodes)-1):		
        f.write('.include diff'+str(x+1)+'.sp\n')
    f.write(".include 'neuron.sp'\n")
    f.write(".option post \n") # Post causes everything to get spit out :)
    f.write('.option ingold=2 artist=2 psf=2\n')
    f.write('.OPTION DELMAX=1NS\n')
    #f.write('.option probe\n')
    #f.write('.probe v(output0) \n')
    f.write(".op\n")
    f.write(".PARAM VddVal=%f\n"%vdd)
    f.write(".PARAM VssVal=%f\n"%vss)
    f.write(".PARAM tsampling=%fn\n"%tsampling)
    for i in range(len(nodes)-1):
        f.write(".include 'layer"+ str(i+1)+".sp'\n")
    for i in range(len(nodes)-1):
        print(f"\nCalling mapLayer.mapLayer for Layer {i} with the following arguments:\n")
        print(f"nodes[i] (input neurons): {nodes[i]}")
        print(f"nodes[i+1] (output neurons): {nodes[i+1]}")
        print(f"layer number (i+1): {i + 1}")
        print(f"hpar[i]: {hpar[i]}")
        print(f"vpar[i]: {vpar[i]}")
        print(f"metal: {metal}")
        print(f"T: {T}")
        print(f"H: {H}")
        print(f"L: {L}")
        print(f"W: {W}")
        print(f"D: {D}")
        print(f"eps: {eps}")
        print(f"rho: {rho}")
        print(f"weight_var: {weight_var}")
        print(f"data_dir: {data_dir}")
        print(f"spice_dir: {spice_dir}\n")

        
        mapLayer.mapLayer(nodes[i],nodes[i+1],i+1,hpar[i],vpar[i],metal,T,H,L,W,D,eps,rho,weight_var,data_dir,spice_dir)
        f.write("Xlayer"+ str(i+1)+" vdd vss 0 ")
        for i2 in range(nodes[i]):
            if (i==0):
                f.write("in%d "%i2)
            else:
                f.write("out%d_%d "%(i,i2))
        for i3 in range(nodes[i+1]):
            if (i==len(nodes)-2):
                f.write("output%d "%i3)
            else:
                f.write("out%d_%d "%(i+1,i3))
        f.write("layer"+ str(i+1)+"\n\n\n")


    f.write("\n\n**********Input Test****************\n\n")
    c=open(data_dir+'/'+'data_sim.txt', "r")
    input_str = c.readlines()[0].split()
    input_num = [float(num) for num in input_str]
    for line in range(nodes[0]):
        f.write("v%d in%d 0 PWL( 0n 0 "%(line,line))
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
			
			
