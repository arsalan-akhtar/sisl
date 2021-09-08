#------------------------------------------------------
# Function for Plotting the NEB results
#-----------------------------------------------------
def plot_barrier(NEBfolder,fig_name,dpi_in ):
    """
    #==========================================================================
    "Plotting NEB"
    #==========================================================================

    """

    # Libray imports
    import os, shutil
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy import interpolate

    print ("*******************************************************************")
    print ("Plotting NEB...! ")
    print ("*******************************************************************")


    NAME_OF_NEB_RESULTS = NEBfolder / 'NEB.results'
    neb_r = np.loadtxt(NAME_OF_NEB_RESULTS)
    Number_of_images = int( neb_r[:,0].max()-1)
    print(f"DEBUG: {Number_of_images}")
    data=[]

    with open(NAME_OF_NEB_RESULTS) as f:
        for ind, line in enumerate(f, 1):
            if ind>2:
                #print (line)
                data.append(line)
    while '\n' in data:
        data.remove('\n')

    image_number=[]
    reaction_coordinates=[]
    Energy=[]
    E_diff=[]
    Curvature=[]
    Max_Force=[]
    for i in range(len(data)):
        image_number.append(float(data[i].split()[0]))
        reaction_coordinates.append(float(data[i].split()[1]))
        Energy.append(float(data[i].split()[2]))
        E_diff.append(float(data[i].split()[3]))
        Curvature.append(float(data[i].split()[4]))
        Max_Force.append(float(data[i].split()[5]))

    Total_Number=Number_of_images+2
    shift=len(E_diff)-Total_Number


    im=[]
    x=[]
    y=[]
    y2=[]
    for i in range(Total_Number):
        im.append(np.array(int(image_number[shift+i])))
        x.append(np.array(reaction_coordinates[shift+i]))
        y.append(np.array(E_diff[shift+i]))
        y2.append(np.array(Energy[shift+i]))

    #Finding Barrier
    Barrier=max(y)




    xnew = np.linspace(0, x[len(x)-1], num=1000, endpoint=True)
    f1=interp1d(x, y,kind='linear')
    f2=interp1d(x, y, kind='cubic')
    f3=interp1d(x, y, kind='quadratic')


    plt.plot(x,y,"o",xnew,f1(xnew),"-",xnew,f2(xnew),"--",xnew,f3(xnew),'r')
    plt.title("Barrier Energy = "+str(Barrier)+" eV")
    plt.legend(['data', 'linear', 'cubic','quadratic'], loc='best')

    plt.savefig(NEBfolder / fig_name.format('.pdf'))
    plt.savefig(NEBfolder / fig_name.format('.png'))
    plt.savefig(NEBfolder / fig_name.format('.jpeg'),dpi = dpi_in)
    #plt.savefig(str(NEBfolder) + str(fig_name) + '.png')
    #plt.savefig(str(NEBfolder) + str(fig_name) + '.pdf')
    #plt.savefig(str(NEBfolder) + str(fig_name) + '.jpeg',dpi=dpi_in)
    #plt.plot(x,y,"o",x,ynew,'+')
    #if Plot == True:
    #    plt.savefig(inputfile +  fig_name + '.png')
    #    plt.savefig(inputfile +  fig_name + '.pdf')
    #    plt.savefig(inputfile +  fig_name + '.jpeg',dpi=dpi_in)
    #else:
    #    plt.show()
    #return plt


def NEB_Fix (path,fdf_name,xyz_input,xyz_file_out):
    """
    """
    import sisl
    import numpy as np
    fdf = sisl.get_sile(path+fdf_name)
    geom = fdf.read_geometry()
    geom_ase = geom.toASE()
    missing = geom_ase.get_chemical_symbols()

    # %%
    temp_xyz =(path + xyz_input)
    f = open(temp_xyz, "r")

    # %%
    coor = np.array([])
    count = 0
    for i in f:
        if count>1:
            #print (i.split())
            coor = np.append(coor,float(i.split()[0]))
            coor = np.append(coor,float(i.split()[1]))
            coor = np.append(coor,float(i.split()[2]))
        count = count + 1
    #lines = f.readlines()
    f.close()

# %%
    coor = coor.reshape(geom.na,3)

# %%
    #missing_symbols = np.array(['',''])
    missing_symbols = np.array(missing)

    # %%
    #missing_symbols= np.append(missing_symbols, missing)

# %%
    missing_symbols= missing_symbols.reshape(geom.na,1)

# %%
    new_xyz_coor = np.hstack((missing_symbols,coor))

# %%

    f = open(xyz_file_out+".xyz", "w")
    f.writelines(str(geom.na)+'\n')
    f.write(xyz_file_out+'\n')
    for item in new_xyz_coor:
        f.writelines(" {}\t {}  {}  {}\n".format(item[0],item[1],item[2],item[3]))
    f.close()

    print ('Done')

def constraint_parameters_block(sisl_image,folder):
    """
    Finding ghosts & Adding Constraint Block

    """
    import sisl
    import numpy as np
    Ghost_Species ={}
    ghost_check = sisl.AtomGhost(1).__class__
    n_ghost = 0
    for i in range(sisl_image.na):
        #print(i)
        if sisl_image.atom[i].__class__ == ghost_check:
            n_ghost = n_ghost + 1
            print (f"Number of Ghost {n_ghost} and index is {i}")
            Ghost_Species [int(sisl_image.atoms.specie[i])+1] = 'species-i'

    F = open(folder/'ghost_block_temp','w')
    F.writelines('\n')
    if len(Ghost_Species) >= 1:
        F.writelines('%block Geometry-Constraints\n')
        for k,v in Ghost_Species.items():
            F.writelines(f"species-i  {k}  \n")
        F.writelines('%endblock Geometry-Constraints\n')
    F.writelines('\n')
    F.writelines('%include parameters.fdf \n')
    F.close()

def file_cat (file_name,file_to_added,file_out):
    """
    # Python program to
    # demonstrate merging
    # of two files
    """
    #print ("DEBUG:",file_name)
    #print ("DEBUG:",file_to_added)
    #print ("DEBUG",file_out)
    data = data2 = ""
    # Reading data from file1
    with open(file_name) as fp:
        data = fp.read()
    # Reading data from file2
    with open(file_to_added) as fp:
        data2 = fp.read()
    # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += data2
    with open (file_out, 'w') as fp:
        fp.write(data)
    
def finding_atom_position_index(geometry,position_to_found,rtol=1e-2,atol=1e-2):
    """
    Read geometry and specific postions coordinates to return the index number of array
    geometry = sisl.xyz
    position_to_found = np.array of atom cartesian position
    rtol 
    atol
    """
    import numpy as np

    for i in range(len(geometry.xyz)):
        if np.isclose(float(geometry.xyz[i][0]),position_to_found[0],rtol,atol) and np.isclose(float(geometry.xyz[i][1]),position_to_found[1],rtol,atol) and np.isclose(float(geometry.xyz[i][2]),position_to_found[2],rtol,atol):
            print (f"The Atom Index is ({i}) and Position is ({position_to_found})")
            index=i
    try:
        return index
    except:
          print(" Couldn't Find the Atom")



def prepare_ase_for_vacancy_exchange(initial_geometry,
                                     initial_vacancy_position_index,
                                     final_vacancy_position_index,
                                     ghost):
    """
    """
    import sisl
    print("DEBUG: Removing Vacancies !")
    print(f"DEBUG: Removing Atom with index ({initial_vacancy_position_index}) for Initial Structure")
    print(f"DEBUG: Removing Atom with index ({final_vacancy_position_index}) for Final Structure")
    if ghost == True:
        # For initial vacancy
        initial_ghost_Z = sisl.AtomGhost(initial_geometry.atoms.Z[initial_vacancy_position_index])
        ghost_initial = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
                                          atoms= sisl.AtomGhost( initial_ghost_Z.Z , tag="V_"+initial_ghost_Z.symbol+"_ghost"))

        # For final vacancy
        final_ghost_Z = sisl.AtomGhost(initial_geometry.atoms.Z[final_vacancy_position_index])
        ghost_final = sisl.Geometry(initial_geometry.xyz[final_vacancy_position_index],
                                          atoms= sisl.AtomGhost( final_ghost_Z.Z , tag="V_"+final_ghost_Z.symbol+"_ghost"))


    trace_atom_initial = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
                                      atoms= initial_geometry.atoms.Z[initial_vacancy_position_index]
                                     )
    trace_atom_final = sisl.Geometry(initial_geometry.xyz[final_vacancy_position_index],
                                    atoms=  initial_geometry.atoms.Z[final_vacancy_position_index]
                                     )

    moving_info = sisl.Atom(initial_geometry.atoms.Z[initial_vacancy_position_index])
    moving_atom = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
                                          atoms= sisl.Atom( moving_info.Z , tag=moving_info.symbol+"_moving"))

  #  moving_atom = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
  #                                        atoms= sisl.Atom( initial_ghost_Z.Z , tag=initial_ghost_Z.symbol+"_moving"))

    initial_for_ASE = initial_geometry.remove(initial_vacancy_position_index)
    final_for_ASE = initial_geometry.remove(final_vacancy_position_index)

    if (final_vacancy_position_index) > (initial_vacancy_position_index):
            print ("Order : Final Atom Position > Initial Atom Position")
            initial_for_ASE = initial_for_ASE.remove(final_vacancy_position_index-1)
            final_for_ASE = final_for_ASE.remove(initial_vacancy_position_index)
    if (final_vacancy_position_index) < (initial_vacancy_position_index):
            print ("Order : Initial Atom Position > Final Atom Position")
            initial_for_ASE = initial_for_ASE.remove(final_vacancy_position_index)
            final_for_ASE = final_for_ASE.remove(initial_vacancy_position_index-1)

    initial_for_ASE = initial_for_ASE.add(trace_atom_final)
    final_for_ASE = final_for_ASE.add(trace_atom_initial)
    initial_for_ASE = initial_for_ASE.toASE()
    final_for_ASE = final_for_ASE.toASE()
    if ghost == True:
        info = {'initial': initial_for_ASE,
                'final' : final_for_ASE,
                'trace_atom_initial' : trace_atom_initial,
                'trace_atom_final' : trace_atom_final,
                'ghost_initial' : ghost_initial,
                'ghost_final' : ghost_final}
    else:
        info = {'initial': initial_for_ASE,
                'final' : final_for_ASE,
                'trace_atom_initial' : trace_atom_initial,
                'trace_atom_final' : trace_atom_final}

    return info

def prepare_ase_for_relaxed(initial_structure,final_structure,ghost):
    """
    """
    import sisl
    if ghost:
        initial_ghost_Z = sisl.AtomGhost(initial_structure.atoms.Z[-2])
        ghost_initial = sisl.Geometry(initial_structure.xyz[-2],
                                          atoms= sisl.AtomGhost( initial_ghost_Z.Z , tag="V_"+initial_ghost_Z.symbol+"_ghost"))

        final_ghost_Z = sisl.AtomGhost(final_structure.atoms.Z[-1])
        ghost_final = sisl.Geometry(final_structure.xyz[-1],
                                          atoms= sisl.AtomGhost( final_ghost_Z.Z , tag="V_"+final_ghost_Z.symbol+"_ghost"))
        initial_for_ASE = initial_structure.remove(-2)
        initial_for_ASE = initial_for_ASE.remove(-1)
        final_for_ASE = final_structure.remove(-2)
        final_for_ASE = final_for_ASE.remove(-1)
        initial_for_ASE = initial_for_ASE.toASE()
        final_for_ASE = final_for_ASE.toASE()

        info = {'initial': initial_for_ASE,
                'final' : final_for_ASE,
                'ghost_initial' : ghost_initial,
                'ghost_final' : ghost_final}
    
    else:
        initial_for_ASE = initial_structure.toASE()
        final_for_ASE = final_structure.toASE()
        info = {'initial': initial_for_ASE,
                'final' : final_for_ASE}

    
    return info 

def prepare_ase_for_interstitial(initial_geometry,
                                 initial_vacancy_position_index,
                                 final_vacancy_position,
                                 ghost):
    """
    """
    import sisl
    print("DEBUG: Removing Vacancies !")
    print(f"DEBUG: Removing Atom with index ({initial_vacancy_position_index}) for Initial Structure")
    print(f"DEBUG: Removing Atom with index ({final_vacancy_position}) for Final Structure")
    if ghost == True:
        # For initial vacancy
        initial_ghost_Z = sisl.AtomGhost(initial_geometry.atoms.Z[initial_vacancy_position_index])
        ghost_initial = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
                                          atoms= sisl.AtomGhost( initial_ghost_Z.Z , tag="V_"+initial_ghost_Z.symbol+"_ghost"))

        # For final vacancy
        final_ghost_Z = sisl.AtomGhost(initial_geometry.atoms.Z[initial_vacancy_position_index])
        #ghost_final = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
        ghost_final = sisl.Geometry(xyz= final_vacancy_position,
                                          atoms= sisl.AtomGhost( final_ghost_Z.Z , tag="V_"+final_ghost_Z.symbol+"_ghost"))


    trace_atom_initial = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
                                      atoms= initial_geometry.atoms.Z[initial_vacancy_position_index]
                                     )
    trace_atom_final = sisl.Geometry(xyz=final_vacancy_position,
                                    atoms=  initial_geometry.atoms.Z[initial_vacancy_position_index]
                                     )
    moving_info = sisl.Atom(initial_geometry.atoms.Z[initial_vacancy_position_index])
    moving_atom = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
                                          atoms= sisl.Atom( moving_info.Z , tag=moving_info.symbol+"_moving"))
                                          #atoms= sisl.Atom( initial_ghost_Z.Z , tag=initial_ghost_Z.symbol+"_moving"))

    initial_for_ASE = initial_geometry.remove(initial_vacancy_position_index)
    final_for_ASE = initial_for_ASE 

    initial_for_ASE = initial_for_ASE.add(trace_atom_initial)
    final_for_ASE = final_for_ASE.add(trace_atom_final)
    initial_for_ASE = initial_for_ASE.toASE()
    final_for_ASE = final_for_ASE.toASE()
    if ghost == True:
        info = {'initial': initial_for_ASE,
                'final' : final_for_ASE,
                'trace_atom_initial' : trace_atom_initial,
                'trace_atom_final' : trace_atom_final,
                'ghost_initial' : ghost_initial,
                'ghost_final' : ghost_final}
    else:
        info = {'initial': initial_for_ASE,
                'final' : final_for_ASE,
                'trace_atom_initial' : trace_atom_initial,
                'trace_atom_final' : trace_atom_final}

    return info



def prepare_ase_for_kick(initial_geometry,
                         initial_vacancy_position_index,
                         final_vacancy_position_index,
                         kicked_atom_final_position,
                         ghost):
    """
    """

    import sisl
    if ghost == True:
        # For initial vacancy
        initial_ghost_Z = sisl.AtomGhost(initial_geometry.atoms.Z[initial_vacancy_position_index])
        ghost_initial = sisl.Geometry(initial_geometry.xyz[initial_vacancy_position_index],
                                          atoms= sisl.AtomGhost( initial_ghost_Z.Z , tag="V_"+initial_ghost_Z.symbol+"_ghost"))

        # For final vacancy
        final_ghost_Z = sisl.AtomGhost(initial_geometry.atoms.Z[final_vacancy_position_index])
        ghost_final = sisl.Geometry(initial_geometry.xyz[final_vacancy_position_index],
        #ghost_final = sisl.Geometry(xyz= initial_geometry.xyz[final_vacancy_position,
                                          atoms= sisl.AtomGhost( final_ghost_Z.Z , tag="V_"+final_ghost_Z.symbol+"_ghost"))


    TraceAtom_A_Initial_Info = sisl.Atom( initial_geometry.atoms.Z[initial_vacancy_position_index])
    TraceAtom_A_Initial = sisl.Geometry(xyz=initial_geometry.xyz[initial_vacancy_position_index],
                                           atoms= sisl.Atom(TraceAtom_A_Initial_Info.Z,tag=TraceAtom_A_Initial_Info.symbol                                                  ))

    TraceAtom_A_Final_Info  = sisl.Atom(initial_geometry.atoms.Z[initial_vacancy_position_index ])
    TraceAtom_A_Final = sisl.Geometry(initial_geometry.xyz[final_vacancy_position_index ],
                                    atoms= sisl.Atom(TraceAtom_A_Final_Info.Z,tag=TraceAtom_A_Final_Info.symbol))

    TraceAtom_B_Initial_Info = sisl.Atom(initial_geometry.atoms.Z[final_vacancy_position_index])
    TraceAtom_B_Initial = sisl.Geometry(initial_geometry.xyz[final_vacancy_position_index],
                                    atoms= sisl.Atom (TraceAtom_B_Initial_Info.Z,tag=TraceAtom_B_Initial_Info.symbol))

    TraceAtom_B_Kicked_Info = sisl.Atom(initial_geometry.atoms.Z[final_vacancy_position_index])
    TraceAtom_B_Kicked = sisl.Geometry(kicked_atom_final_position,
                                    atoms= sisl.Atom(TraceAtom_B_Kicked_Info.Z,tag=TraceAtom_B_Kicked_Info.symbol))


    initial_for_ASE = initial_geometry.remove(initial_vacancy_position_index)
    final_for_ASE = initial_geometry.remove(final_vacancy_position_index)

    if final_vacancy_position_index > initial_vacancy_position_index :
        print ("Order : Final Atom Position > Initial Atom Position")
        initial_for_ASE = initial_for_ASE.remove(final_vacancy_position_index-1)
        final_for_ASE = final_for_ASE.remove(initial_vacancy_position_index)
    if final_vacancy_position_index  < initial_vacancy_position_index :
        print ("Order : Initial Atom Position > Final Atom Position")
        initial_for_ASE = initial_for_ASE.remove(final_vacancy_position_index)
        final_for_ASE = final_for_ASE.remove(initial_vacancy_position_index-1)

    initial_for_ASE = initial_for_ASE.add(TraceAtom_A_Initial)
    final_for_ASE = final_for_ASE.add(TraceAtom_A_Final)
    initial_for_ASE = initial_for_ASE.toASE()
    final_for_ASE = final_for_ASE.toASE()
    if ghost:
        info = {'initial': initial_for_ASE,
            'final':final_for_ASE,
            'trace_atom_A_initial' : TraceAtom_A_Initial,
            'trace_atom_A_final' : TraceAtom_A_Final,
            'trace_atom_B_initial' : TraceAtom_B_Initial,
            'trace_atom_B_kicked' : TraceAtom_B_Kicked,
            'ghost_initial' : ghost_initial,
            'ghost_final' : ghost_final}
                 



    else:
        info = {'initial': initial_for_ASE,
            'final':final_for_ASE,
            'trace_atom_A_initial' : TraceAtom_A_Initial,
            'trace_atom_A_final' : TraceAtom_A_Final,
            'trace_atom_B_initial' : TraceAtom_B_Initial,
            'trace_atom_B_kicked' : TraceAtom_B_Kicked,
                 }


    return info
            

