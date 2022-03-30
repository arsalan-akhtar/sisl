from __future__ import absolute_import

#from SiestaSurfaces.SiestaSurfacesBase   import SiestaSurfacesBase
from SiestaSurfaces.surface_generator.surface_generator import SlabMaker

class SurfaceGeneratorWorkChain():

    """
    Class To handle the SLAB Generation
    """
    def __init__(self,
                bulk_structure ,
                surface_direction ,
                number_of_layers ,
                vacuum_axis ,
                vacuum_amount ,
                debug_slab = False
                ):

        self.bulk_structure = bulk_structure
        self.surface_direction = surface_direction
        self.number_of_layers = number_of_layers
        self.vacuum_axis = vacuum_axis
        self.vacuum_amount = vacuum_amount 
        self.debug_slab = debug_slab
        
        print("SurfaceGeneratorWorkChain initialized ...")

    def set_debug_slab(self,debug_slab=True):
        self.debug_slab = debug_slab


    def is_debug_slab(self):
        """
        
        """
        if self.debug_slab == True :
            print("DEBUG: Writing Slab For Double Check")
            return True
        else:
            return False


    def Generate_Slab(self):
        """
        Genrate Slab images
        """

    
        print("Generating Slab with miller index {}".format(self.surface_direction))
        print("Generating Slab for {} layers ".format(self.number_of_layers))
        print("Generating Slab for axis {}  :".format( self.vacuum_axis))
        print("Generating Slab with vacuum {} Ang :".format(self.vacuum_amount))


        self.structure = SlabMaker( Bulk = self.bulk_structure ,
                                        direction = self.surface_direction,
                                        nlayers = self.number_of_layers,
                                        axis_of_vacuum = self.vacuum_axis,
                                        vacuum_amount = self.vacuum_amount )
        
        #self.report("DEBUG: The Initial Structure : "+str(self.ctx.initial_structure.sites))

        print("Slab Generated ...!")
        

