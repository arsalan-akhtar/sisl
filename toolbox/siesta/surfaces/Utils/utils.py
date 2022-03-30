#------------------
#
#-------------------

def ASE2Siesta(A):
    """
    Getting ASE Atom Object and Converted to sisl
    
    INPUT:
    ======
    A : ASE Structure

    OUTPUT:
    =======
    sisl structure object
    
    """
    import sisl
    geo=sisl.Geometry.fromASE(A)
    return geo


def Siesta2ASE(path,label,read_from):
    """
    Getting XV Path Object and Converted ASE Atom 
    
    INPUTS:
    =======
    path      : folder path to the file
    label     : file label/name
    read_from : read from either XV or fdf file
    
    OUTPUTS:
    ========
    ase structure object
    """
    import sisl
    
    if read_from == "xv":
        xvfile = sisl.get_sile(path+"/"+label+".XV")
        geometry = xvfile.read_geometry()
    if read_from == "fdf":
        fdffile = sisl.get_sile(path+"/"+"input.fdf")
        geometry = fdffile.read_geometry()
    
    ase_geom = sisl.Geometry.toASE(geometry)
    
    return ase_geom
 
def FindingSurfaceAtomIndex(SislStructure,vaccum_axis):
    """
    Read SislStructure & vaccum axis return the surface_index
    of Atoms in top & bottom layer
    
    INPUTS:
    =======

    SislStructure :  Sisl Structure
    vaccum_axis   :  

    OUTPUTS:
    ========
    surface index

    """
    from sisl import Atom
    import numpy as np
    
    #Sorted_SileStrucutre = X.sort(axis=vaccum_axis)
    SislStructure
    Top_Atom = np.amax(SislStructure.xyz ,axis=(0))[vaccum_axis]
    Bottom_Atom = np.amin(SislStructure.xyz ,axis=(0))[vaccum_axis]
    surface_index =np.array([])
    for i in range(SislStructure.na):
        # Searching for Top
        if np.isclose(SislStructure.xyz[i][2],Top_Atom):
            surface_index = np.append(surface_index,i)

        
        # Searching for Bottom
        if np.isclose(SislStructure.xyz[i][2],Bottom_Atom):
            surface_index = np.append(surface_index,i)
    
    print("Top/Bottom Atom index : {} ".format(surface_index))
    return surface_index


def AddingSurfaceSpeciesSuffix(ASEStructure,SurfaceIndex,SuffixName="surface"):
    """
    Read ASE structure & index and name for index return the different name for Species

    INPUTS:
    =======

    ASEStructure : ase structure object
    SurfaceIndex : index of surface atom
    SuffixName   : name for surface specie "will catcatinate with _ to atom label" 
    default is "surface" 
    Ex.  for O --------> O_surface in siesta fdf 

    OUTPUT:
    =======

    return Species list Tag with provided ASE Structure Object to use for FixingSislWithIndex function!

    """
    Species = ASEStructure.get_chemical_symbols()
    for i in SurfaceIndex:
        print("Adding Suffix ({}) For Atom Index :({}) and Label :({}) ".format(str(SuffixName), int(i),Species[int(i)]))
        Species[int(i)] = Species[int(i)]+"_"+SuffixName
    
    return Species

def FixingSislWithIndex(SislStructure,SpeciesfromASE):
    """ 
    Reading Sisl Structure Object and Species list to tag the name 
    to use for the cases that have different Species like Ghost , Surface , etc...

    INPUTS:
    =======
    SislStructure  : sisl Structure object 
    SpeciesfromASE : provided Species list to tag

    OUTPUT:
    =======
    Sisl Structure with tags which write different speices 
    """
    from sisl import Geometry,Atom
    for i in range(SislStructure.na):
        AtomIndex = Atom (SislStructure.atoms[i].Z,tag=SpeciesfromASE[i])
    #for i in range(SislStructure.na):
        if i ==0:
            Test = Geometry(xyz=SislStructure.xyz[i],
                        atoms=Atom(AtomIndex.Z,tag=AtomIndex.tag),
                        sc=SislStructure.cell
                           )
        if i>0:
            Test += Geometry(xyz=SislStructure.xyz[i],
                        atoms=Atom(AtomIndex.Z,tag=AtomIndex.tag),
                        sc=SislStructure.cell
                            )
    return Test


def FixingSislWithIndex_BUG_TODELETE(SislStructure,SpeciesfromASE):
    """ 
    Reading Sisl Structure Object and Species list to tag the name 
    to use for the cases that have different Species like Ghost , Surface , etc...

    INPUTS:
    =======
    SislStructure  : sisl Structure object 
    SpeciesfromASE : provided Species list to tag

    OUTPUT:
    =======
    Sisl Structure with tags which write different speices 
    """
    from sisl import Geometry,Atom
    for i in range(SislStructure.na):
        AtomIndex = Atom (SislStructure.atoms.Z,tag=SpeciesfromASE)
    for i in range(SislStructure.na):
        if i ==0:
            Test = Geometry(xyz=SislStructure.xyz[i],
                        atoms=Atom(AtomIndex.Z[i],tag=AtomIndex.tag[i]),
                        sc=SislStructure.cell
                           )
        if i>0:
            Test += Geometry(xyz=SislStructure.xyz[i],
                        atoms=Atom(AtomIndex.Z[i],tag=AtomIndex.tag[i]),
                        sc=SislStructure.cell
                            )
    return Test


def AllIndexesReturn_PyMatgen(SislStructure_First,SislStructure_Second):
    """
    SislStructure_First  : n layer Structure
    SislStructure_Second : either 2n+1 or 2n layer
    """
    import numpy as np
    Indexes={}

    CenterIndexes = np.array([],dtype=int)
    for j in range(SislStructure_First.na):
        for i in range(SislStructure_Second.na):

            if (np.allclose(SislStructure_First.xyz[j], SislStructure_Second.xyz[i])):
                print(i,np.allclose(SislStructure_First.xyz[j], SislStructure_Second.xyz[i]))
                CenterIndexes = np.append(CenterIndexes,i)

    AllIndexes = np.array(range(0,SislStructure_Second.na))

    GhostIndexes = np.delete(AllIndexes,CenterIndexes)
    SurfaceGhostsIndex = GhostIndexes
    print ("Surface Indexes: {}".format(SurfaceGhostsIndex))
    print ("Center Indexes: {}".format(CenterIndexes))
    print ("Ghost Indexes: {}".format(GhostIndexes))
    Indexes['All'] = AllIndexes
    Indexes['Surface'] = SurfaceGhostsIndex
    Indexes['Center'] = CenterIndexes
    Indexes['Ghost'] = GhostIndexes

    return Indexes

def AllIndexesReturn(SislStructure_First,SislStructure_Second):

    """
    SislStructure_First  : n layer Structure
    SislStructure_Second : either 2n+1 or 2n layer
    """
    import numpy as np
    Indexes={}
    res=np.isin(SislStructure_Second.xyz, SislStructure_First.xyz)
    SurfaceGhostsIndex = np.array([])
    for j in range(SislStructure_Second.na):
        if res[j].all():
            #print (j,"No")
            pass
        else:
            #print(j,"No")
            SurfaceGhostsIndex = np.append(SurfaceGhostsIndex,j)
    AllIndexes = np.array(range(0,SislStructure_Second.na))
    CenterIndexes=np.array([],dtype=int)
    for i in range(SislStructure_First.na,2*SislStructure_First.na):
        #print (i)
        CenterIndexes = np.append(CenterIndexes,int(i))
    GhostIndexes = np.delete(AllIndexes,CenterIndexes)

    print ("Surface Indexes: {}".format(SurfaceGhostsIndex))
    print ("Center Indexes: {}".format(CenterIndexes))
    print ("Ghost Indexes: {}".format(GhostIndexes))
    Indexes['All'] = AllIndexes
    Indexes['Surface'] = SurfaceGhostsIndex
    Indexes['Center'] = CenterIndexes
    Indexes['Ghost'] = GhostIndexes
    return Indexes



def EachLayerIndexes (vacuum_axis,Indexes,Structure_ASE):
    """

    """
    from sisl import Geometry,Atom
    import numpy as np

    Structure_ASE_3 = Structure_ASE
    CenterIndexes = Indexes["Center"]
    AllIndexes = Indexes["All"]
    GhostIndexes = Indexes["Ghost"]

    vacuum_axis_atoms =np.array([])
    #for i in range(SileStructure_3.na):
    #    vacuum_axis_atoms = np.append(vacuum_axis_atoms,SileStructure_3.xyz[i][vacuum_axis])
    for i in range(Structure_ASE_3.get_global_number_of_atoms()):
        vacuum_axis_atoms = np.append(vacuum_axis_atoms,Structure_ASE_3.get_positions()[i][vacuum_axis])
        
    vacuum_axis_atoms = np.unique(vacuum_axis_atoms)

    layer_ghost = {}
    layer_center = {}
    layer_all = {}
    for j in range(len(vacuum_axis_atoms)):
        for l in range(len(CenterIndexes)):
            if Structure_ASE_3.get_positions()[CenterIndexes[l]][vacuum_axis]==vacuum_axis_atoms[j]:
                t_center = Geometry(xyz=Structure_ASE_3.get_positions()[CenterIndexes[l]],
                                    atoms=Atom(Structure_ASE_3.get_atomic_numbers()[CenterIndexes[l]]),
                                    sc=Structure_ASE_3.cell)
                layer_center.setdefault(j, []).append(t_center)

        for k in range(len(AllIndexes)):
            if Structure_ASE_3.get_positions()[AllIndexes[k]][vacuum_axis]==vacuum_axis_atoms[j]:
                t_all = Geometry(xyz=Structure_ASE_3.get_positions()[AllIndexes[k]],
                                 atoms=Atom(Structure_ASE_3.get_atomic_numbers()[AllIndexes[k]]),
                                 sc=Structure_ASE_3.cell)
                layer_all.setdefault(j, []).append(t_all)
        for i in range(len(GhostIndexes)):
            if Structure_ASE_3.get_positions()[GhostIndexes[i]][vacuum_axis]==vacuum_axis_atoms[j]:
                print (i,Structure_ASE_3.get_positions()[GhostIndexes[i]],Structure_ASE_3.get_atomic_numbers()[GhostIndexes[i]])
                t = Geometry(xyz=Structure_ASE_3.get_positions()[GhostIndexes[i]],
                             atoms=Atom(Structure_ASE_3.get_atomic_numbers()[GhostIndexes[i]]),
                             sc=Structure_ASE_3.cell)
                layer_ghost.setdefault(j, []).append(t)
    layer_dict = {}

    layer_dict ['all']= layer_all
    layer_dict ['center']= layer_center
    layer_dict ['ghost']= layer_ghost
    return layer_dict



def CenterStructure(layer_center,Structure_ASE_3):
    """

    """
    from sisl import Geometry,Atom
    start = 0
    count = 0
    for key in layer_center.keys():
        #print (key,range(len(layer_center[key])),start)
        for l in range(len(layer_center[key])):
            count +=1
            #print(l,layer_center[key][l].xyz,start)
            if start == 0:
                print (l,"no",start,layer_center[key][l].xyz)
                New = Geometry(xyz=layer_center[key][l].xyz,
                           atoms=Atom(layer_center[key][l].atoms.Z[0]),
                           sc=Structure_ASE_3.cell)
            if start!=0 :
                    print (l,"yes",start,layer_center[key][l].xyz)
                    New += Geometry(xyz=layer_center[key][l].xyz,
                               atoms=Atom(layer_center[key][l].atoms.Z[0]),
                               sc=Structure_ASE_3.cell)
            start += 1
    print (count)
    return New


def AvailableTopBottomLayerIndex(Ldict):
    """
    Returning all top & bottom layer indexes
    """
    def top_bottom_index(layer_ghost):

        l_c = 0
        index_bottom = 0
        index_top = 0
        for i in layer_ghost.keys():
            #print (i,l_c)
            if i -(l_c) == 0:
                #print("No" ,i-(l_c+1))
                index_bottom +=1
            else:
                #print(i,l_c,"yess")
                index_top = i
                break
            l_c =l_c+ 1
        index_bottom = index_bottom-1

        index ={}
        index ['top']=index_top
        index ['bottom']=index_bottom
        return index

    top_indx = top_bottom_index(Ldict['ghost'])['top']
    bottom_idx = top_bottom_index(Ldict['ghost'])['bottom']
    #========================
    toplayer_parition = {}
    bottomlayer_parition = {}
    tp_i = 0
    for i in Ldict['ghost'].keys():
        #if i > 12:
        if i >=top_indx:
            tp_i += 1
            toplayer_parition[tp_i] = i
    #bt_i = 5
    bt_i = bottom_idx
    for i in Ldict['ghost'].keys():
        if 0<=i<=5:
            bottomlayer_parition[i+1] = bt_i
            bt_i -=1
            #print (i)
    layer_partition = {}
    layer_partition['top'] = toplayer_parition
    layer_partition['bottom'] = bottomlayer_parition

    #print (")
    return layer_partition


def PuttingBack(FromTopLayers,FromBottomLayers,TopBottomL):
    """
    
    """
    import numpy as np
    PutBack ={}
    Top = np.array([],dtype=int)
    Bottom = np.array([],dtype=int)

    for t in range(1,FromTopLayers+1):
        print("Top Atoms {} {}".format(t,TopBottomL['top'][t]))
        Top = np.append(Top,TopBottomL['top'][t])
    for bt in range(1,FromBottomLayers+1):
        print("Bottom Atoms {}{}".format(bt,TopBottomL['bottom'][bt]))
        Bottom = np.append(Bottom,TopBottomL['bottom'][bt])

    PutBack['bottom'] = Bottom
    PutBack['top'] = Top
    
    return PutBack


def AddGhostsToSurface(Structure_in,layer,top,bottom,name_index="ghost"):
    """

    """
    from sisl import Geometry,Atom
    layer_center = layer['center']
    layer_ghost  = layer['ghost']

    #top = str(top)
    #bottom = str(bottom)
    New_ghost = Structure_in
    # For Top
    for i in top:
        print ("Putting Back From Top Layer {}".format(i))
        for l in range(len(layer_ghost[i])):
            New_ghost += Geometry(xyz=layer_ghost[i][l].xyz,
                        atoms=Atom(-1*layer_ghost[i][l].atoms.Z[0],tag=layer_ghost[i][l].toASE().get_chemical_symbols()[0] +"_"+name_index),
                       )

    # For Bottom
    for k in bottom:
        print ("Putting Back From Bottom Layer {}".format(k))
        for l in range(len(layer_ghost[k])):
            New_ghost += Geometry(xyz=layer_ghost[k][l].xyz,
                        atoms=Atom(-1*layer_ghost[k][l].atoms.Z[0],tag=layer_ghost[k][l].toASE().get_chemical_symbols()[0] +"_"+name_index),
                       )

    return New_ghost



def AreaofSlab(SislStructure,vacuum_axis):
    """
    Calculating Area with normal plan vector along vacuum axis
    """
    import numpy as np
    if vacuum_axis == 0:
        a_cross_b = np.cross(SislStructure.cell[1],SislStructure.cell[2])
        plane = "y-z"
    if vacuum_axis == 1:
        a_cross_b = np.cross(SislStructure.cell[0],SislStructure.cell[2])
        plane = "x-z"
    if vacuum_axis == 2:
        a_cross_b = np.cross(SislStructure.cell[0],SislStructure.cell[1])
        plane = "x-y"
    Area = np.linalg.norm(a_cross_b)
    print("The Area of plane '{}' is {}".format(plane,Area))
    return Area


def SurfaceEnergyStoichiometricSymmetric(BulkEnergy,SlabEnergy,nlayer,Area):
    """
    Calculating Surface Energy for Stoichiometric and Symmetric Slabs

    INPUTS:
    =======

    BulkEnergy: Bulk energy
    SlabEnergy: Slab energy
    nlayer    : # of layer
    Area      : Area of Slab along vaccum direction

    OUTPUT:
    =======
    Surface Energy
    """
    SurfaceEnergy = ((SlabEnergy - (nlayer * BulkEnergy)) / (2*Area))

    print ("Surface energy for '{}' layer is '{}' [eV/Ang**2]".format(nlayer,SurfaceEnergy))
    return SurfaceEnergy

