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



class SurfacesBase:
    """
    The base class to compute the different images for neb
    
    host_path                    : Path of Calculations
    host_structure               :
    ghost                        :
    relaxed                      :
    """
    def __init__(self,
                 bulk_path = None,
                 bulk_fdf_name = None,
                 surface_direction = None,
                 vacuum_axis = None,
                 vacuum_amount = None,
                 number_of_layers = 15,
                 ghost = False,
                 ):

        self.bulk_path = bulk_path
        self.bulk_fdf_name = bulk_fdf_name
        self.surface_direction = surface_direction
        self.vacuum_axis = vacuum_axis
        self.vacuum_amount = vacuum_amount
        self.number_of_layers = number_of_layers
        self.ghost = ghost
        
        #--------------------------------------------------------------------------------------
        #            To Test
        #--------------------------------------------------------------------------------------

        
        #self.Surface_Generation = SurfaceGeneratorWorkChain(bulk_structure =  self.bulk_structure,
        #self.Surface = SurfaceGeneratorWorkChain(bulk_structure =  self.bulk_structure,
        #                                          surface_direction = self.surface_direction,
        #                                          number_of_layers = self.number_of_layers,
        #                                          vacuum_axis = self.vacuum_axis,
        #                                          vacuum_amount =  self.vacuum_amount
        #        )

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
        from surface_generator.surface_generator_wk  import (SurfaceGeneratorWorkChain)
        
        #self.Surface_Generation = SurfaceGeneratorWorkChain(bulk_structure =  self.bulk_structure,
        self.Surface = SurfaceGeneratorWorkChain(bulk_structure =  self.bulk_structure,
                                                  surface_direction = self.surface_direction,
                                                  number_of_layers = self.number_of_layers,
                                                  vacuum_axis = self.vacuum_axis,
                                                  vacuum_amount =  self.vacuum_amount
                )
      
    #----------------
    # Generating Slab
    #----------------

    def Surface(self):
        """
        Generating Slab
        """
        from surface_generator.surface_generator_wk  import (SurfaceGeneratorWorkChain)
        
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
                 bulk_path = None,
                 bulk_output_name = None,
                 bulk_structure = None,
                 slab_structures = None,
                 pseudos_path = None,
                 flos_path = None,
                 surface_direction = None,
                 vacuum_axis = None,
                 vacuum_amount = None,
                 relaxed_bulk_structure = None,
                 ghost = None,
                 flos_file_name_relax = None,
                 relax_engine = 'lua',
                 relaxed = None,
                 results_folders = None,
                 bulk_input_name = None,
                 number_of_layers = 20, # None,
                 number_of_ghost_layer_top = 1,
                 number_of_ghost_layer_bottom = 1,
                 type_of_surface_species = "diffuse",
                 name_of_surface_species = "surface"
                 ):

        self.bulk_input_name = bulk_input_name 
        self.bulk_output_name = bulk_output_name
        self.type_of_surface_species = type_of_surface_species
        self.name_of_surface_species = name_of_surface_species
        self.number_of_ghost_layer_top = number_of_ghost_layer_top
        self.number_of_ghost_layer_bottom = number_of_ghost_layer_bottom
       
        super().__init__(
                         bulk_path,
                         bulk_input_name,
                         bulk_output_name,
                         bulk_structure,
                         slab_structures,
                         pseudos_path,
                         flos_path,
                         surface_direction,
                         vacuum_axis,
                         vacuum_amount,
                         relaxed_bulk_structure,
                         results_folders,
                         number_of_layers,
                         flos_file_name_relax,
                         relax_engine ,
                         relaxed,
                         ghost,
                         )
        #self.relax_engine = relax_engine
        

        #self.bulk_fdf_name = bulk_fdf_name 
        #self.number_of_layers = self.number_of_layers 

        
        #super(SiestaSurfacesBase,self).__init__(number_of_layers)
    
        #self.bulk_path = bulk_path
        #self.bulk_structure = bulk_structure
        #self.number_of_layers = self.Siesta_SurfaceBase.number_of_layers 
        #self.surface_direction = surface_direction
        #self.vacuum_axis = vacuum_axis
        #self.vacuum_amount = vacuum_amount 
        #self.ghost = ghost
        #self.type_of_surface_species = type_of_surface_species
        #self.name_of_surface_species = name_of_surface_species

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

    def Generate_Slabs(self):
        """ 
        Setup the Surface workchain

        # Check if barrier scheme is valid:
        """
        import sisl
        
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

        self.Siesta_Surface = {}
        self.Siesta_Surface_ghosts = {}

        self.Siesta_Surface_SISL = {}

        if  self.type_of_surface_species == "diffuse" or self.type_of_surface_species == "none":
            print (" ========================================================================")
            print (" Generating Scheme '{}' For  SIESTA ".format(self.type_of_surface_species))
            print (" ========================================================================")
            for i in range(1,self.number_of_layers+1):
                print (i)
                self.Siesta_Surface [str(i)] = SiestaSurfacesBase(bulk_structure = ase,
                                number_of_layers = i, #self.number_of_layers,
                                surface_direction = self.surface_direction,
                                vacuum_axis = self.vacuum_axis,
                                vacuum_amount = self.vacuum_amount
                )
                self.Siesta_Surface[str(i)].Surface.Generate_Slab()  
       


                self.SislStructure = sisl.Geometry.fromASE(self.Siesta_Surface[str(i)].Surface.structure)
    
                SurfaceIndexes = FindingSurfaceAtomIndex(self.SislStructure,2)
                Species = AddingSurfaceSpeciesSuffix(self.Siesta_Surface[str(i)].Surface.structure,SurfaceIndexes,self.name_of_surface_species)    
            
                if self.type_of_surface_species == "diffuse":
                    self.Siesta_Surface_SISL [i] = FixingSislWithIndex(self.SislStructure,Species)
                if self.type_of_surface_species == "none":
                    self.Siesta_Surface_SISL [i] = self.SislStructure

        if self.type_of_surface_species == "ghost-odd":
            print (" ========================================================================")
            print (" Generating Scheme '{}' For  SIESTA ".format(self.type_of_surface_species))
            print (" ========================================================================")
            #self.raise_not_implemented()
            for i in range(1,2*(self.number_of_layers)+2):
                print (i)
                self.Siesta_Surface_ghosts [str(i)] = SiestaSurfacesBase(bulk_structure = ase,
                                number_of_layers = i, #self.number_of_layers,
                                surface_direction = self.surface_direction,
                                vacuum_axis = self.vacuum_axis,
                                vacuum_amount = self.vacuum_amount
                )
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


        #--------------------------------------
        # Passing to IO Class
        #--------------------------------------

        self.IO = SiestaSurfacesIO( Siesta_Surface_SISL = self.Siesta_Surface_SISL,
                                   vacuum_axis= self.vacuum_axis,
                                   bulk_path= self.bulk_path,
                number_of_layers = self.number_of_layers,
                surface_direction = self.surface_direction,
                vacuum_amount = self.vacuum_amount,
                flos_path = self.flos_path,
                flos_file_name_relax = self.flos_file_name_relax,
                relax_engine = self.relax_engine,
                type_of_surface_species = self.type_of_surface_species
                #ghost = self.ghost
                )
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
        self.IO = SiestaSurfacesIO( #Siesta_Surface_SISL = self.Siesta_Surface_SISL,
                                   vacuum_axis= self.vacuum_axis,
                                   bulk_path= self.bulk_path,
                number_of_layers = self.number_of_layers,
                surface_direction = self.surface_direction,
                vacuum_amount = self.vacuum_amount,
                flos_path = self.flos_path,
                flos_file_name_relax = self.flos_file_name_relax,
                relax_engine = self.relax_engine,
                type_of_surface_species = self.type_of_surface_species,
                surface_energies = self.SurfaceEnergies
                )


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



        

