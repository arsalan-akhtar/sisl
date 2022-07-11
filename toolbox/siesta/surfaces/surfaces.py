########################################################################################
# Copyright (c), The Siesta Group. All rights reserved.                                #
#                                                                                      #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

#from SiestaSurfaces.Utils.utils_siesta import print_siesta_fdf,read_siesta_fdf
from .surface_generator.surface_generator_wk  import (SurfaceGeneratorWorkChain)

#from SiestaSurfaces.SiestaSurfacesIO import SiestaSurfacesIO
#from SiestaSurfaces.Utils.utils_siesta import read_siesta_fdf,read_siesta_XV

from .Utils.utils import AddingSurfaceSpeciesSuffix,FindingSurfaceAtomIndex,FixingSislWithIndex
from .Utils.utils import AllIndexesReturn
from .Utils.utils import EachLayerIndexes
from .Utils.utils import  CenterStructure
from .Utils.utils import AvailableTopBottomLayerIndex
from .Utils.utils import AddGhostsToSurface,PuttingBack
import numpy as np
import sisl
import sys

class SurfacesBase:
    """
    The base class to compute the different images for neb
    
    host_path                    : Path of Calculations
    host_structure               :
    ghost                        :
    relaxed                      :
    """
    def __init__(self,
                 bulk_structure = None,
                 surface_direction = None,
                 vacuum_axis = None,
                 vacuum_amount = 30.0,
                 number_of_layers = 15,
                 ghost = False,
                 ):

        self.bulk_structure = bulk_structure
        self.surface_direction = surface_direction
        self.vacuum_axis = vacuum_axis
        self.vacuum_amount = vacuum_amount
        self.number_of_layers = number_of_layers
        self.ghost = ghost
        

    #----------------
    # Generating Inforamtion
    #----------------


    def Setup_SurfaceBase(self):
        """ 
        Setup the Surface workchain

        # Check if barrier scheme is valid:
        """
        if self.vacuum_axis == 0:
            va = 'x'
        if self.vacuum_axis == 1:
            va = 'y'
        if self.vacuum_axis == 2:
            va = 'z'
        
        print("Checking Surface Type ...")
        print ("The Slab Direction {}".format(self.surface_direction))
        print("The Slab Axis of Vacuum {} : {} direction ".format(self.vacuum_axis,va))
        print("The Slab Amoun of Axis {}".format(self.vacuum_amount))
        
        #--------------------------------------------------------------------------------------
        #            To Test
        #--------------------------------------------------------------------------------------
        #from surface_generator.surface_generator_wk  import (SurfaceGeneratorWorkChain)
        
        #self.Surface_Generation = SurfaceGeneratorWorkChain(bulk_structure =  self.bulk_structure,
        #self.Surface = SurfaceGeneratorWorkChain(bulk_structure =  self.bulk_structure,
        #                                          surface_direction = self.surface_direction,
        #                                          number_of_layers = self.number_of_layers,
        #                                          vacuum_axis = self.vacuum_axis,
        #                                          vacuum_amount =  self.vacuum_amount
        #        )
      
    #----------------
    # Generating Slab
    #----------------

    def Surface(self):
        """
        Generating Slab
        """
        #from surface_generator.surface_generator_wk  import (SurfaceGeneratorWorkChain)
        print("DEBUG: SurfaceBase Class") 
        #self.Surface_Generation = SurfaceGeneratorWorkChain(bulk_structure =  self.bulk_structure,
        self.Surface = SurfaceGeneratorWorkChain(bulk_structure =  self.bulk_structure,
                                                  surface_direction = self.surface_direction,
                                                  number_of_layers = self.number_of_layers,
                                                  vacuum_axis = self.vacuum_axis,
                                                  vacuum_amount =  self.vacuum_amount
                )
        #self.Surface_Structure = self.Surface_Generation.Generate_Slab()



    #============================================
    # Compute Barrier Corrections
    #============================================    
    
    def compute_surface_energy(self):
        """ 
        Compute the Barrier Energy 
        """
        #self.out("surface_energy", self.ctx.surface_energy)    
        pass

    def raise_not_implemented (self):
        """
        """

        print("Not Implemented...")
        sys.exit() 
#=========================================================================
#
#
#
#=========================================================================
# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-SIESTA-LUA-NEB authors. All rights reserved.                #
#                                                                                      #
# AiiDA-SIESTA-LUA-Defects is hosted on GitHub at https:                               #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################

class Surfaces(SurfacesBase):
    """

    """
    def __init__(self,
                 bulk_structure = None,
                 bulk_path = None,
                 bulk_input_name = None,
                 surface_direction = [1,0,0],
                 vacuum_axis = 2,
                 vacuum_amount =30.0 ,
                 ghost = None,
                 bulk_output_name = None,
                 slab_structures = None,
                 relaxed_bulk_structure = None,
                 relaxed = None,
                 results_folders = None,
                 number_of_layers = 20, # None,
                 number_of_ghost_layer_top = 1,
                 number_of_ghost_layer_bottom = 1,
                 type_of_surface_species = "diffuse",
                 name_of_surface_species = "surface",
                 oxidation_state_dict = None,
                 slab_centered = True,
                 ):

        self.bulk_input_name = bulk_input_name 
        self.bulk_output_name = bulk_output_name
        self.slab_structures = slab_structures
        self.relaxed_bulk_structure = relaxed_bulk_structure
        self.relaxed = relaxed
        self.results_folders = results_folders
        self.number_of_layers = number_of_layers
        self.number_of_ghost_layer_top = number_of_ghost_layer_top
        self.number_of_ghost_layer_bottom = number_of_ghost_layer_bottom
        self.type_of_surface_species = type_of_surface_species
        self.name_of_surface_species = name_of_surface_species
        self.oxidation_state_dict = oxidation_state_dict      
        self.slab_centered = slab_centered
        super().__init__(
                 bulk_structure = bulk_structure,
                 surface_direction = surface_direction,
                 vacuum_axis = vacuum_axis,
                 vacuum_amount = vacuum_amount ,
                 number_of_layers = number_of_layers,
                 ghost = ghost,
                         )

    #-------------
    # Set Methods
    #-------------

    def set_bulk_fdf_name(self,host_fdf_name):
        """
        """
        self.host_fdf_name = host_fdf_name
 
    #def set_results_folder(self,results_folders):
    #    """
    #    """
    #
    #    self.results_folders = results_folders

    def Genarate_Slabs_Pymatgen(self):
        """
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core.surface import SlabGenerator 
        from pymatgen.core.surface import generate_all_slabs

        print ("Generating Slabs Using PyMatgen Lib")
        self._Pymatgen_Structure = AseAtomsAdaptor.get_structure(self.bulk_structure.toASE())
        if self.oxidation_state_dict is not None:
            self._Pymatgen_Structure.add_oxidation_state_by_element(self.oxidation_state_dict)
        else:
            print("Cannot find polar structures... Continue")
        self._slab = generate_all_slabs (structure = self._Pymatgen_Structure,
                                              max_index = 1,
                                              min_slab_size = self.number_of_layers ,
                                              min_vacuum_size = self.vacuum_amount ,
                                              max_normal_search = True,
                                              center_slab = self.slab_centered)


    def Genarate_Slabs_Pymatgen_all_nl_NEW(self,trial_nl=35):
        """
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core.surface import generate_all_slabs
        print ("Generating Slabs Using PyMatgen Lib")


        self._Pymatgen_Structure = AseAtomsAdaptor.get_structure(self.bulk_structure.toASE())
        self.slabs_pymatgen = {}
        self.slabs_ase = {}
        self.slabs_sisl = {}
        #nl = trail
        if self.oxidation_state_dict is not None:
            self._Pymatgen_Structure.add_oxidation_state_by_element(self.oxidation_state_dict)
        else:
            print("Cannot find polar structures... Continue")
        layer = 1
        self.slab_surface_available = []
        slabgen_all = generate_all_slabs (structure = self._Pymatgen_Structure,
                                  max_index = 1,
                                  min_slab_size = 1 ,
                                  min_vacuum_size = self.vacuum_amount ,
                                  max_normal_search = True,
                                  center_slab = self.slab_centered)
        for n, slab in enumerate(slabgen_all):
            if slab.is_symmetric():
                sym = "symmetric"
            else:
                sym = "nosymmetric"
            if slab.is_polar():
                pol = "polar"
            else:
                pol = "nonpolar"
            k = str(slab.miller_index[0])+str(slab.miller_index[1])+str(slab.miller_index[2])+"-"+sym+"-"+pol
            print(k)

            self.slab_surface_available.append(k)
            name = "1-"+k
            self.slabs_pymatgen[name] = slab
            self.slabs_ase [name]= AseAtomsAdaptor.get_atoms(self.slabs_pymatgen[name])
            self.slabs_sisl [name]= sisl.Geometry.fromASE(self.slabs_ase[name])

        for li in self.slab_surface_available: #key_in:
            try_nl = 1
            nl = trial_nl #45
            l=1
            while try_nl <= nl:
                print("=========================")
                print(f"Trying nl : {try_nl,l} ...")
                print("=========================")
                slabgen_all = generate_all_slabs (structure = self._Pymatgen_Structure,
                                          max_index = 1,
                                          min_slab_size = try_nl ,
                                          min_vacuum_size = self.vacuum_amount ,
                                          max_normal_search = True,
                                          center_slab = self.slab_centered)
                for n, slab in enumerate(slabgen_all):
                    if slab.is_symmetric():
                        sym = "symmetric"
                    else:
                        sym = "nosymmetric"
                    if slab.is_polar():
                        pol = "polar"
                    else:
                        pol = "nonpolar"
                    k = str(slab.miller_index[0])+str(slab.miller_index[1])+str(slab.miller_index[2])+"-"+sym+"-"+pol        
                    #if slab_temp[0] == k:
                    if li == k:
                        print(f"For : {k}")
                        dk_previous = str(l)+"-"+k
                        print(f"DK Previous is {dk_previous}")
                        print(f"Checking :{slab.num_sites} > {self.slabs_pymatgen[dk_previous].num_sites}")
                        if slab.num_sites > self.slabs_pymatgen[dk_previous].num_sites:
                            l=l+1
                            dk_next = str(l)+"-"+k
                            print(f"Found for {l}.......")  
                            print(f"DK Next is {dk_next}")
                            self.slabs_pymatgen[dk_next] = slab 
                            self.slabs_ase [dk_next]= AseAtomsAdaptor.get_atoms(self.slabs_pymatgen[dk_next])
                            self.slabs_sisl [dk_next]= sisl.Geometry.fromASE(self.slabs_ase[dk_next])
                            #break        

                    try_nl=try_nl+ 1            
        print (f"Found until ==========> {l}")




    def Generate_Slabs_Pymatgen_all_nl(self):
        """
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core.surface import SlabGenerator 
        from pymatgen.core.surface import generate_all_slabs

        print ("Generating Slabs Using PyMatgen Lib")
        self.slabs_pymatgen = {}
        self.slabs_ase = {}
        self.slabs_sisl = {}

        self._Pymatgen_Structure = AseAtomsAdaptor.get_structure(self.bulk_structure.toASE())
        if self.oxidation_state_dict is not None:
            self._Pymatgen_Structure.add_oxidation_state_by_element(self.oxidation_state_dict)
        else:
            print("Cannot find polar structures... Continue")
 
        for nl in range(1,self.number_of_layers+1):

            self._slabgen_all = generate_all_slabs (structure = self._Pymatgen_Structure,
                                              max_index = 1,
                                              min_slab_size = nl ,
                                              min_vacuum_size = self.vacuum_amount ,
                                              max_normal_search = True,
                                              center_slab = self.slab_centered)
            for n, slab in enumerate(self._slabgen_all):
                if slab.is_symmetric():
                    sym = "symmetric"
                else:
                    sym = "nosymmetric"
                if slab.is_polar():
                    pol = "polar"
                else:
                    pol = "nonpolar"
                print(f"{n}\n",
                      f"miller index:{slab.miller_index}\n "
                      f"is Symmetric:{slab.is_symmetric()}\n" ,
                      f"is Polar {slab.is_polar()}\n")
                k = str(nl)+"-"+str(slab.miller_index[0])+str(slab.miller_index[1])+str(slab.miller_index[2])+"-"+sym+"-"+pol
                print(k)

            
                self.slabs_pymatgen [k]= slab
                self.slabs_ase [k]= AseAtomsAdaptor.get_atoms(self.slabs_pymatgen[k])
                self.slabs_sisl [k]= sisl.Geometry.fromASE(self.slabs_ase[k])

    def Fixing_sisl_diffuse(self):
        """
        Adding diffuse for Siesta Species in FDF
        """

        from toolbox.siesta.surfaces.Utils.utils import AddingSurfaceSpeciesSuffix
        from toolbox.siesta.surfaces.Utils.utils import FindingSurfaceAtomIndex
        from toolbox.siesta.surfaces.Utils.utils import FixingSislWithIndex
    
        self.Siesta_Surface_SISL = {}
        for k,v in self.slabs_sisl.items():
            print(k)
            SurfaceIndexes = FindingSurfaceAtomIndex(self.slabs_sisl[k],2)
            Species = AddingSurfaceSpeciesSuffix(ASEStructure = self.slabs_ase[k],
                                                 SurfaceIndex = SurfaceIndexes,
                                                 SuffixName = self.name_of_surface_species)

            #if self.type_of_surface_species == "diffuse":
            #self.Siesta_Surface_SISL [k] = FixingSislWithIndex(self.slabs_sisl[k],Species)
            self.slabs_sisl [k] = FixingSislWithIndex(self.slabs_sisl[k],Species)

        #return self.Siesta_Surface_SISL 
        #return self.slabs_sisl

    def Generate_Slabs_ASE(self):
        """ 
        Setup the Surface workchain

        # Check if barrier scheme is valid:
        """
        import sisl
        print("*********************************")
        print("NOTE : its Better to use Genarate_Slabs_Pymatgen_all_nl_NEW ")

        if self.bulk_input_name is not None:
            if 'fdf' in self.bulk_input_name:
                Geo = read_siesta_fdf(self.bulk_path,
                              self.bulk_input_name)
            if 'XV' in self.bulk_input_name:
                Geo = read_siesta_XV(self.bulk_path,
                                  self.bulk_input_name)
        

            #if self.bulk_relaxed:
            #Geo = read_siesta_fdf(self.bulk_path,
            #                      self.bulk_fdf_name)
            #Geo = read_siesta_XV(self.bulk_path,
            #                      self.bulk_fdf_name)
            ase = sisl.Geometry.toASE(Geo["Geometry"])
        else:
            ase = self.bulk_structure.toASE()

        self.Siesta_Surface = {}
        self.Siesta_Surface_ghosts = {}
    
        self.Siesta_Surface_SISL = {}

        if  self.type_of_surface_species == "diffuse" or self.type_of_surface_species == "none":
            print (" ========================================================================")
            print (" Generating Scheme '{}' For  SIESTA ".format(self.type_of_surface_species))
            print (" ========================================================================")
            for i in range(1,self.number_of_layers+1):
                print (i)
                self.Siesta_Surface [str(i)] = SurfacesBase(bulk_structure = ase,
                                number_of_layers = i, #self.number_of_layers,
                                surface_direction = self.surface_direction,
                                vacuum_axis = self.vacuum_axis,
                                vacuum_amount = self.vacuum_amount
                )
                #self.Siesta_Surface[str(i)].Surface.Generate_Slab()
                self.Siesta_Surface[str(i)].Surface()
                self.Siesta_Surface[str(i)].Surface.Generate_Slab()  
       


                self.SislStructure = sisl.Geometry.fromASE(self.Siesta_Surface[str(i)].Surface.structure)
    
                SurfaceIndexes = FindingSurfaceAtomIndex(self.SislStructure,2)
                Species = AddingSurfaceSpeciesSuffix(self.Siesta_Surface[str(i)].Surface.structure,SurfaceIndexes,self.name_of_surface_species)    
            
                if self.type_of_surface_species == "diffuse":
                    self.Siesta_Surface_SISL [i] = FixingSislWithIndex(self.SislStructure,Species)
                if self.type_of_surface_species == "none":
                    self.Siesta_Surface_SISL [i] = self.SislStructure

            self.slabs_sisl = self.Siesta_Surface_SISL

        if self.type_of_surface_species == "ghost-odd":
            print (" ========================================================================")
            print (" Generating Scheme '{}' For  SIESTA ".format(self.type_of_surface_species))
            print (" ========================================================================")
            #self.raise_not_implemented()
            for i in range(1,2*(self.number_of_layers)+2):
                print (i)
                self.Siesta_Surface_ghosts [str(i)] = SurfacesBase(bulk_structure = ase,
                                number_of_layers = i, #self.number_of_layers,
                                surface_direction = self.surface_direction,
                                vacuum_axis = self.vacuum_axis,
                                vacuum_amount = self.vacuum_amount
                )
                self.Siesta_Surface_ghosts[str(i)].Surface()
                self.Siesta_Surface_ghosts[str(i)].Surface.Generate_Slab() 
             
            for i in range(1,self.number_of_layers+1):
                print (i,2*i+1)
                Structure_ASE_1 = self.Siesta_Surface_ghosts [str(i)].Surface.structure
                Structure_ASE_3 = self.Siesta_Surface_ghosts [str(2*(i)+1)].Surface.structure

                SileStructure_1 = sisl.Geometry.fromASE(Structure_ASE_1)
                SileStructure_3 = sisl.Geometry.fromASE(Structure_ASE_3)
                SurfaceIndexes_1 = FindingSurfaceAtomIndex(SileStructure_1,self.vacuum_axis)
                SurfaceIndexes_3 = FindingSurfaceAtomIndex(SileStructure_3,self.vacuum_axis)
                #Species_1 = AddingSurfaceSpeciesSuffix(Structure_ASE_1,SurfaceIndexes_1,"ghost")
                #Species_3 = AddingSurfaceSpeciesSuffix(Structure_ASE_3,SurfaceIndexes_3,"ghost")
                Index_dict=AllIndexesReturn(SileStructure_1,SileStructure_3)
                Ldict=EachLayerIndexes (vacuum_axis=self.vacuum_axis,Indexes=Index_dict,Structure_ASE=Structure_ASE_3)
                New = CenterStructure(Ldict['center'],Structure_ASE_3)
                self.AvailTopBottomL=AvailableTopBottomLayerIndex(Ldict)
                PutBack = PuttingBack(self.number_of_ghost_layer_top,self.number_of_ghost_layer_bottom,self.AvailTopBottomL)
                self.Siesta_Surface_SISL [i] = AddGhostsToSurface(New,Ldict,PutBack['top'],PutBack['bottom'])

            self.slabs_sisl = self.Siesta_Surface_SISL 
        #--------------------------------------
        # Passing to IO Class
        #--------------------------------------

        #self.IO = SiestaSurfacesIO( Siesta_Surface_SISL = self.Siesta_Surface_SISL,
        #                           vacuum_axis= self.vacuum_axis,
        #                           bulk_path= self.bulk_path,
        #        number_of_layers = self.number_of_layers,
        #        surface_direction = self.surface_direction,
        #        vacuum_amount = self.vacuum_amount,
        #        flos_path = self.flos_path,
        #        flos_file_name_relax = self.flos_file_name_relax,
        #        relax_engine = self.relax_engine,
        #        type_of_surface_species = self.type_of_surface_species
                #ghost = self.ghost
        #        )
        #self.IO.set_vacuum

    
    def Prepare_Slab_results(self):

        """
        """
        from SiestaSurfaces.Utils import utils_defects
        
        BulkEnergy  = utils_defects.output_energy_manual(self.bulk_path,self.bulk_output_name)
        print ("Bulk Energy is {}".format(BulkEnergy))
        
        from SiestaSurfaces.surface_energy.surface_energies import SurfaceEnergiesWorkChain


        A = SurfaceEnergiesWorkChain(bulk_energy=BulkEnergy,
                           number_of_layers=self.number_of_layers,
                           vacuum_axis=self.vacuum_axis,
                           results_folders = self.results_folders
                        )
        
        A.CalculateSurfaceEnergies()
        self.SurfaceEnergies = A.surface_energy
        
        #self.IO = SiestaSurfacesIO( #Siesta_Surface_SISL = self.Siesta_Surface_SISL,
        #                           vacuum_axis= self.vacuum_axis,
        #                           bulk_path= self.bulk_path,
        #        number_of_layers = self.number_of_layers,
        #        surface_direction = self.surface_direction,
        #        vacuum_amount = self.vacuum_amount,
        #        flos_path = self.flos_path,
        #        flos_file_name_relax = self.flos_file_name_relax,
        #        relax_engine = self.relax_engine,
        #        type_of_surface_species = self.type_of_surface_species,
        #        surface_energies = self.SurfaceEnergies
        #        )


    def plot(self,ylim=[1.0,-1.0]):
            """
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.interpolate import interp1d
            from scipy import interpolate

            x=np.linspace(0 ,len(self.SurfaceEnergies),num=self.number_of_layers)

            y =self.SurfaceEnergies
            xnew = np.linspace(0, x*[x-1], num=1000, endpoint=True)
            f1=interp1d(x, y,kind='linear')
            

            plt.plot(x,y,"o",x,f1(x),"-")
            plt.legend(['data', 'linear', ], loc='best')
            plt.title("Surface Energy {}  ".format(str(self.surface_direction[0]) +
                str(self.surface_direction[1]) +
                str(self.surface_direction[2]) + "  $eV/Ang^{2}$"
                ))
            plt.legend(['data', 'linear'], loc='best')
            plt.ylim(ylim[0],ylim[1])
            #plt.plot(x,y,"o")



        
    def write_slabs(self,which_type='sisl',output_type = "XSF",folder_name='./slabs'):
        """
        """
        import pathlib , os
        Folder = pathlib.Path(folder_name)
        if Folder.exists():
            print("Folder Exist ... Pass")
        else:
            os.mkdir(Folder)
            for k,v in self.slabs_sisl.items():
                print(k)
                file_name = str(k)
                if which_type == "sisl":
                    compelete_name = file_name+"."+output_type
                    self.slabs_sisl[k].write(Folder/compelete_name)

    def prepare_slab_relaxation(self,json_info_filename="slabs_info"):
        """
        """
        import pathlib , os
        import json
        for_json = []
        for k,v in self.slabs_sisl.items():

            Folder_name = pathlib.Path(k)
            if Folder_name.exists():
                print(f"Folder Exist for {k} ... Pass")
            else:
                for_json.append(k)
                print(k)
                os.mkdir(Folder_name)
                name = Folder_name/"input.fdf"
                self.slabs_sisl[k].write(name)
        js_name = f"{json_info_filename}.json"
        with open(js_name,"w") as outfile:
            json.dump(for_json,outfile,indent=1)


    def read_result_info_dir(self,folder_path,result_path,json_info_file="slabs_info"):
        """
        """
        import json
        import pathlib
        slabs_dir_info = {}
        with open (json_info_file,"r") as jsonfile:
            data = json.load(jsonfile)
        for i in data:
            full_path = f"{folder_path}/{i}/{result_path}"
            print(full_path)
            slabs_dir_info[i] = pathlib.Path(full_path)
        return slabs_dir_info
