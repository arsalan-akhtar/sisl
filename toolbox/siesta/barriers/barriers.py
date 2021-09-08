# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The SislSiestaBarriers authors. All rights reserved.                  #
# SislSiestaBarriers is hosted on GitHub at :                                          #
# https://github.com/zerothi/sisl/toolbox/siesta/barriers                              #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
import pathlib
import shutil , os
from .utils_barriers import plot_barrier,constraint_parameters_block,file_cat
from .utils_barriers import finding_atom_position_index
from .utils_barriers import prepare_ase_for_relaxed
from .utils_barriers import prepare_ase_for_vacancy_exchange
from .utils_barriers import prepare_ase_for_interstitial 
from .utils_barriers import prepare_ase_for_kick
from sisl import Geometry
import sisl
import numpy as np
__author__ = "Arsalan Akhatar"
__copyright__ = "Copyright 2021, SIESTA Group"
__version__ = "0.1"
__maintainer__ = "Arsalan Akhtar"
__email__ = "arsalan_akhtar@outlook.com" 
__status__ = "Development"
__date__ = "Janurary 30, 2021"


class SiestaBarriersBaseNick:
    """
    The base class to compute the different images for neb
    
    Parameters
    ----------
    images : list of Geometry objects
       the images (including relaxed, first, and final image, last).
    path : callable or pathlib.Path or str
       path to write geometry to.
       If a callable it should have the following arguments ``image, index, total``
       where ``image`` corresponds to ``images[index]`` and ``total`` is the length of `images`.
       If a `pathlib.Path` it will be appended ``_{index}_{total}`` for checking indices.
       If a `str` it may contain formatting rules with ``{index}`` or ``{total}``.
       Internally ``self.path`` will be a callable with the above arguments.
    engine : handler for NEB calculation

    Examples
    --------

    I don't think you should implement this class to do the interpolation, what if the
    user wants to try some them-selves, it might be easier to provide wrappers for functions
    that interpolates, and then returns the full thing.
    It becomes much simpler and easier for the end user to fit their needs.

    Generally classes should do as little as possible, and preferably one thing only.
    Your classes are simply too difficult to understand for a new user. Is my bet.
    I am suggesting major changes here since I think that is required if this is to be used.
    Sorry to be this blunt.
    
    It also isn't clear to me the exact procedure or scope of these classes.
    Are you only focusing on using these scripts for the Lua engines? Or would
    you imagine later to extend with Python back-ends (say I-Pi?).

    I don't think you should have an IO class as well. It might be a set of
    functions, but otherwise it might be useful *in* the class if needed.

    """

    #def __init__(self, neb_path , images , path='image'):
    def __init__(self, 
                 neb_path =None , 
                 images = None , 
                 path='image',
                 initial_relaxed_path = None,
                 final_relaxed_path = None,
                 initial_relaxed_fdf_name = None,
                 final_relaxed_fdf_name = None,
                 ):
        # Store all images in the class, convert to sisl.Geometry
        self.images = images
        print(self.images)
        if self.images is not [None]:
            self.images = [Geometry.new(image) for image in images]
            # we need to have at least initial and final
            # While it doesn't make sense to calculate barriers for
            # two points, it might be useful for setting up initial and final
            # geometries in specific directories.
            #assert len(self.images) >= 2
            assert len(self.images) >= 1

        if isinstance(path, str):
            path = pathlib.Path(path)
        if isinstance(neb_path,str):
            neb_path = pathlib.Path(neb_path)

        if callable(path):
            def _path(image, index, total):
                return pathlib.Path(path(image, index, total))
        elif isinstance(path, pathlib.Path):
            # Convert to func
            def _path(image, index, total):
                #return path.with_suffix(path.suffix + f"._{index}_{total}")
                return path.with_suffix(path.suffix + f".{index}_{total}")
        else:
            raise ValueError("Unknown argument type for 'path' not one of [callable, str, Path]")
        self.path = _path
        self.neb_path = neb_path
        self.fdf_image = None
       
        #TEST ----------------------------------------------------------------
        self.initial_relaxed_path = initial_relaxed_path
        self.final_relaxed_path = final_relaxed_path
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name
        self.final_relaxed_fdf_name = final_relaxed_fdf_name
        print("DEBUG:",self.initial_relaxed_path,self.final_relaxed_path,self.initial_relaxed_fdf_name,final_relaxed_fdf_name)
        #TEST ----------------------------------------------------------------

    def __len__(self):
        """ Number of images (excluding initial and final) """
        return len(self.images) - 2

    @property
    def initial(self):
        return self.images[0]

    @property
    def final(self):
        return self.images[-1]


    def _prepare_flags(self, files, overwrite):
        total = len(self.images)
        if isinstance(files, str):
            files = [files]
        if isinstance(overwrite, bool):
            # ensure a value per image
            overwrite = [overwrite for _ in range(total)]
        return files, overwrite

    def prepare(self, files='image.xyz', overwrite=False):
        """ Prepare the NEB calculation. This will create all the folders and geometries.

        Parameters
        ----------
        files : str or list of str
           write the image files in the ``self.path`` returned directory.
           If a list, all suffixes will be written.
           The directory will be created if not existing.
           The `files` argument may contain formatted strings ``{index}`` or
           ``{total}`` will be accessible.
        overwrite : bool or list of bool, optional
           whether to overwrite any existing files. Optionally a flag per image.
        """
        # generally you should try and avoid using os.chdir
        # It will cause you more pain than actual gain. ;)
        # Using relative paths are way more versatile and powerful
        total = len(self.images)
        files, overwrite = self._prepare_flags(files, overwrite)
        assert len(overwrite) == total

        for index, (overwrite, image) in enumerate(zip(overwrite, self.images)):
            path = self.path(image, index, total)
            path.mkdir(parents=True, exist_ok=True)
            print("DEBUG:",path)
            # prepare the directory
            for file in files:
                print("DEBUG:",file)
                file = path / file.format(index=index, total=total)
                if overwrite or not file.is_file():
                    # now write geometry
                    image.write(file)
                    

    # if you are not going to use the `set_*` methods, then don't add them.
    # If you have a set you should generally use them in your __init__ call
    # to ensure any setup is done correctly.
    # Also, there seemed to be quite a bit of inconsistency there.
    # I.e. you could setup everything, then change the number of images?
    # This does not make sense and it might be much better to have a simpler
    # class that is easier to maintain.
    def check_images(self,image_folder='images',image_name = 'image', image_format = 'xsf',overwrite=True):
        """
        """

        total = len(self.images)
        image_folder = pathlib.Path(image_folder)
        print("DEBUG:",image_folder)
        image_folder.mkdir(parents=True,exist_ok = overwrite)

        #files, overwrite = self._prepare_flags(files, overwrite)
        files, overwrite = self._prepare_flags(image_name, overwrite)
        print("DEBUG:",files)
        print("DEBUG:",overwrite)

        #assert len(overwrite) == total
        for index, (overwrite, image) in enumerate(zip(overwrite, self.images)):
            print("DEBUG:",index,overwrite)
            file_name = [image_name +'-' + str(index)+'.'+image_format]
            print ("DEBUG:",file_name)
            for file in file_name:
                file = image_folder / file.format(index=index,total=total)
                print("DEBUG:",file)
                if overwrite or not file.is_file():
                    # now write geometry
                    image.write(file)


    def prepare_neb(self,image_name='image',fdf_name='input',overwrite=False):
        """
        """
        total = len(self.images)
        neb_path = self.neb_path 
        print("DEBUG:",neb_path)
        neb_path.mkdir(parents=True,exist_ok = overwrite)

        #files, overwrite = self._prepare_flags(files, overwrite)
        files, overwrite = self._prepare_flags(image_name, overwrite)
        print("DEBUG:",files)
        print("DEBUG:",overwrite)
        
        #assert len(overwrite) == total
        for index, (overwrite, image) in enumerate(zip(overwrite, self.images)):
            print("DEBUG:",index,overwrite)
            #file_name = ['image-' + str(index)+'.xyz']
            file_name = [image_name +'-' + str(index)+'.xyz']
            print ("DEBUG:",file_name)
            for file in file_name:
                file = neb_path / file.format(index=index,total=total)
                print("DEBUG:",file)
                if overwrite or not file.is_file():
                    # now write geometry
                    image.write(file)
       
        fdf_file_name = [fdf_name+'.fdf']
        image = self.fdf_image
        for file in fdf_file_name:
            file = neb_path / file.format(index = index, total = total)
            print("DEBUG:",file)
            image.write(file)
            constraint_parameters_block(image,neb_path)
            file_cat(file , neb_path/'ghost_block_temp',file )
            os.remove(neb_path/'ghost_block_temp')

        # Copying DM files
        if sisl.get_sile(self.initial_relaxed_path/self.initial_relaxed_fdf_name).get("SystemLabel"):
            initial_systemlabel = sisl.get_sile(self.initial_relaxed_path/self.initial_relaxed_fdf_name).get("SystemLabel")+".DM"
        else:
            initial_systemlabel = 'siesta.DM'
        if sisl.get_sile(self.final_relaxed_path/self.final_relaxed_fdf_name).get("SystemLabel"):
            final_systemlabel = sisl.get_sile(self.final_relaxed_path/self.final_relaxed_fdf_name).get("SystemLabel")+".DM"
        else:
            final_systemlabel = 'siesta.DM'

        print(f"DEBUG: Initial SystemLable is {initial_systemlabel}")
        print(f"DEBUG: Final SystemLable is {final_systemlabel}")
        shutil.copy(self.initial_relaxed_path/initial_systemlabel,neb_path/'NEB.DM.0')
        shutil.copy(self.final_relaxed_path/final_systemlabel,neb_path/f'NEB.DM.{self.number_of_images+1}')
        
    
    
    def plot_neb_results(self):
        """
        """
        neb_results_path = self.neb_path / neb_results_folder_name
        print("DEBUG",neb_results_path)
        if neb_results_path.exists():
            plot_barrier(neb_results_path,self.number_of_images,fig_name='NEB',dpi_in=600)
        else:
            raise FileNotFoundError("The Folder doesn't Exists")

    @staticmethod
    def plot_neb_results_from_file(neb_results_folder_name=''):
        """
        """
        
        neb_results_path = pathlib.Path( neb_results_folder_name)
        print("DEBUG",neb_results_path)
        if neb_results_path.exists():
            plot_barrier(neb_results_path,fig_name='NEB',dpi_in=600)
        else:
            raise FileNotFoundError("The Folder doesn't Exists")

    def analyse_neb_image(self,image_n,image_name = 'image', analyse_path = 'image-analyse',neb_results_folder_name = ''):
        """
        image_n : Number
        """
        neb_results_path = self.neb_path / neb_results_folder_name
        print("DEBUG",neb_results_path)
        analyse_folder_name = pathlib.Path(analyse_path+"-"+str(image_n)+"i")
        analyse_folder_name.mkdir(parents=True,exist_ok =False)
        image = image_name + "-"+ str(image_n)+".xyz"
        shutil.copy(neb_results_path/ image ,analyse_folder_name)
        shutil.copy(neb_results_path/ 'input.fdf' ,analyse_folder_name)
        shutil.copy(neb_results_path/ 'parameters.fdf' ,analyse_folder_name)

    
    def prepare_endpoint_relaxation(self):
        """
        """
        self.initial_relaxed_path = pathlib.Path('image-0')
        self.final_relaxed_path = pathlib.Path(f'image-{self.number_of_images+1}')
        self.initial_relaxed_path.mkdir()
        self._initial.write(self.initial_relaxed_path/self.initial_relaxed_fdf_name )
        constraint_parameters_block(self._initial,self.initial_relaxed_path)
        file_cat(self.initial_relaxed_path /self.initial_relaxed_fdf_name ,
                 self.initial_relaxed_path /'ghost_block_temp',
                 self.initial_relaxed_path/self.initial_relaxed_fdf_name  )
        os.remove(self.initial_relaxed_path /'ghost_block_temp')

        self.final_relaxed_path.mkdir()
        self._final.write(self.final_relaxed_path/self.final_relaxed_fdf_name )
        constraint_parameters_block(self._final,self.final_relaxed_path)
        file_cat(self.final_relaxed_path /self.final_relaxed_fdf_name ,
                 self.final_relaxed_path /'ghost_block_temp',
                 self.final_relaxed_path/self.final_relaxed_fdf_name  )
        os.remove(self.final_relaxed_path /'ghost_block_temp')


# The ManualNEB is simple the same as the base class (now)
# It may need some adjustments later.

class ManualNEB(SiestaBarriersBaseNick):
    """
    """
    def __init__(self,
                 initial_structure = None,
                 final_structure = None,
                 number_of_images = None,
                 interpolation_method = 'idpp' ,
                 neb_path= 'neb' ,
                 path='image',
                 initial_relaxed_path = '',
                 final_relaxed_path = '',
                 initial_relaxed_fdf_name = 'input.fdf',
                 final_relaxed_fdf_name = 'input.fdf',

            ):
    #def __init__(self,initial_structure,final_structure,number_of_images,interpolation_method  ,neb_path= 'neb' ,path='image' ):
    #def __init__(self,initial_structure,final_structure,number_of_images,interpolation_method ,path='image' ):

        super().__init__(neb_path = neb_path, 
                         path='image',
                         images = [initial_structure],
                         initial_relaxed_path = pathlib.Path(initial_relaxed_path),
                         final_relaxed_path = pathlib.Path(final_relaxed_path),
                         initial_relaxed_fdf_name = initial_relaxed_fdf_name,
                         final_relaxed_fdf_name = final_relaxed_fdf_name
                         )

        self.initial_structure = initial_structure 
        self.final_structure = final_structure 
        self.number_of_images = number_of_images
        self.interpolation_method = interpolation_method
    
    def prepare_manual_images(self):
        """
        """
        
        from ase.neb import NEB
        import sisl
        initial = sisl.Geometry.toASE(self.initial_structure)
        final = sisl.Geometry.toASE(self.final_structure)
        self.fdf_image = self.initial_structure
        images_ASE = [initial]
        print ("Copying ASE For NEB Image  0 (initial)")
        for i in range(self.number_of_images):
            print ("Copying ASE For NEB Image ",i+1)
            images_ASE.append(initial.copy())
        images_ASE.append(final)
        print ("Copying ASE For NEB Image ",i+2, 'final')
        self.neb = NEB(images_ASE)
        self.neb.interpolate(self.interpolation_method,mic=True)
       
        self.images = []
        for i in range(self.number_of_images+2):
            self.images.append(sisl.Geometry.fromASE(images_ASE[i]))

# It isn't clear at all how a user should use your scripts.
# Perhaps I should refrain from commenting more and you should
# ask a co-worker to run through it. I would highly suggest you
# make it *simpler* since it is too complicated to use.
# Possibly also ask Pol about some suggestions to make it simpler.

# You have lots of Utils.utils_*.py files.
# Instead, put everything that belongs to one method in 1 file.
# This is much simpler and is easier to figure out when things goes
# wrong.
# Also, you seem to have lots of duplicate code around? Why?
# I.e. utils_exchange.py and utils_interstitial.py?
# It really makes the code hard to follow ;)
# Could you also give examples of when the fractional vs. cartesian coordinates
# are useful? The way you check for fractional coordinates is not optimal, if useful at all.
# Why not force users to always use cartesian coordinates?

# I only think you should use CamelCase for classes.
# Methods should be lower_case_like_this (my opinion, makes it easier to
# remember method names).


class LuaNEB(SiestaBarriersBaseNick):
    # this class should implemnet the Lua stuff with copying files etc.
    def __init__(self, lua_scripts, images, *args, **kwargs):
        super().__init__(images, *args, **kwargs)
        if isinstance(lua_scripts, (str, Path)):
            lua_scripts = [lua_scripts]
        self.lua_scripts = [Path(lua_script) for lua_script in lua_scripts]

    def prepare(self, files='image.xyz', overwrite=False):
        super().prepare(files, overwrite)
        _, overwrite = self._prepare_flags(files, overwrite)

        for index, (overwrite, image) in enumerate(zip(overwrite, self.images)):
            path = self.path(image, index, total)
            for lua_script in self.lua_scripts:
                out = path / lua_script.name
                if overwrite or not out.is_file():
                    shutil.copy(lua_script, out)
    

class VacancyExchangeNEB(SiestaBarriersBaseNick):
    """
    """
    #from .utils_barriers import finding_atom_position_index, prepare_ase_for_vacancy_exchange
    def __init__(self,
            pristine_structure,
            initial_vacancy_position,
            final_vacancy_position,
            number_of_images,
            interpolation_method,
            initial_relaxed_path = '',
            final_relaxed_path = '',
            initial_relaxed_fdf_name = 'input.fdf',
            final_relaxed_fdf_name = 'input.fdf',
            ghost = True,
            moving = False,
            relaxed = False,
            neb_path = 'neb',
            path = 'image' ):

        super().__init__(neb_path = neb_path, 
                         path='image',
                         images = [pristine_structure],
                         initial_relaxed_path = pathlib.Path(initial_relaxed_path),
                         final_relaxed_path = pathlib.Path(final_relaxed_path),
                         initial_relaxed_fdf_name = initial_relaxed_fdf_name,
                         final_relaxed_fdf_name = final_relaxed_fdf_name
                         )

        self.pristine_structure = pristine_structure
        self.initial_vacancy_position = initial_vacancy_position
        self.final_vacancy_position = final_vacancy_position 
        self.number_of_images = number_of_images
        self.interpolation_method = interpolation_method
        #self.initial_relaxed_path = pathlib.Path(initial_relaxed_path)
        #self.final_relaxed_path = pathlib.Path(final_relaxed_path)
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name
        self.final_relaxed_fdf_name = final_relaxed_fdf_name
        self.ghost = ghost
        self.moving = moving
        self.relaxed = relaxed



        if self.relaxed == True:
            print ("=================================================")
            print ("The Relaxed Vacancy Exchange Image Generation ...")
            print ("=================================================")
            if not self.initial_relaxed_path.exists()  or not self.final_relaxed_path.exists()  :
                raise FileNotFoundError ("intial/final relaxed path not provided")
            if not (self.initial_relaxed_path/self.initial_relaxed_fdf_name).exists()  or not (self.final_relaxed_path/self.final_relaxed_fdf_name).exists() :
                raise FileNotFoundError("intial/final relaxed fdf not provided")
            self.initial_structure = sisl.get_sile(self.initial_relaxed_path/ self.initial_relaxed_fdf_name).read_geometry(output=True)
            self.final_structure = sisl.get_sile(self.final_relaxed_path/ self.final_relaxed_fdf_name).read_geometry(output=True)
            self.__info = prepare_ase_for_relaxed(self.initial_structure,self.final_structure,self.ghost)
            self.__initial  = self.__info['initial']
            self.__final  = self.__info['final']
        
        else:
             print ("=================================================")
             print ("The Initial Vacancy Exchange Image Generation ...")
             print ("=================================================")
             self.__i_index = finding_atom_position_index(self.pristine_structure , self.initial_vacancy_position)
             self.__f_index = finding_atom_position_index (self.pristine_structure , self.final_vacancy_position)
             self.__info = prepare_ase_for_vacancy_exchange(self.pristine_structure, self.__i_index , self.__f_index,self.ghost)
             self.__initial  = self.__info['initial']
             self.__final  = self.__info['final']
    

    def prepare_vacancy_exchange_images(self):
        """
        """
        
        from ase.neb import NEB
        import sisl

        images_ASE = [self.__initial]
        print ("Copying ASE For NEB Image  0 (initial)")
        for i in range(self.number_of_images):
            print ("Copying ASE For NEB Image ",i+1)
            images_ASE.append(self.__initial.copy())
        images_ASE.append(self.__final)
        print ("Copying ASE For NEB Image ",i+2, 'final')
        self.neb = NEB(images_ASE)
        self.neb.interpolate(self.interpolation_method,mic=True)
       
        self.images = []
        for i in range(self.number_of_images+2):
            self.images.append(sisl.Geometry.fromASE(images_ASE[i]))

        if self.ghost == True: #and self.relaxed is not True:
            for i in range(self.number_of_images+2):
                print(f" Adding Ghost for image {i} in Sisl Geometry Object for initial ")
                self.images[i] = self.images[i].add(self.__info['ghost_initial'])
                print(f" Adding Ghost for image {i} in Sisl Geometry Object for final ")
                self.images[i] = self.images[i].add(self.__info['ghost_final'])

        # This is may be! a new feature, i am still testing it though!
        if self.moving == True and self.relaxed is not True:
             for i in range(self.number_of_images+2):
                if self.ghost:
                    self.images[i].atom[-3] = sisl.Atom(Z = self.images[i].atom[-3].Z,tag=self.images[i].atoms[-3].symbol+"_moving")
                else:
                    self.images[i].atom[-1] = sisl.Atom(Z = self.images[i].atom[-1].Z,tag=self.images[i].atoms[-1].symbol+"_moving")
            
        if self.relaxed is not True:
            self.fdf_image = self.images[0]
            self._initial = self.images[0]
            self._final = self.images[-1]
    
        if self.relaxed :
            self.fdf_image = self.initial_structure


class InterstitialNEB(SiestaBarriersBaseNick):
    """
    """
    def __init__(self,
            pristine_structure,
            initial_atom_position,
            final_atom_position,
            number_of_images,
            interpolation_method = 'idpp',
            initial_relaxed_path = '',
            final_relaxed_path = '',
            initial_relaxed_fdf_name = 'input.fdf',
            final_relaxed_fdf_name = 'input.fdf',
            ghost = False,
            moving = False,
            relaxed = False,
            neb_path = 'neb',
            path = 'image' ):

        super().__init__(neb_path = neb_path,
                         path='image',
                         images = [pristine_structure],
                         initial_relaxed_path = pathlib.Path(initial_relaxed_path),
                         final_relaxed_path = pathlib.Path(final_relaxed_path),
                         initial_relaxed_fdf_name = initial_relaxed_fdf_name,
                         final_relaxed_fdf_name = final_relaxed_fdf_name
                         )

        self.pristine_structure = pristine_structure
        self.initial_atom_position = initial_atom_position
        self.final_atom_position = final_atom_position
        self.number_of_images = number_of_images
        self.interpolation_method = interpolation_method
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name
        self.final_relaxed_fdf_name = final_relaxed_fdf_name
        self.ghost = ghost
        self.moving = moving
        self.relaxed = relaxed
        
        if self.ghost :
            print("########################################################")
            print("                       Warning ...!                     ")
            print("           You are setting ghost flag to (True)         ")
            print("   For the Interstitial it Might give WEIRD results     ")
            print("########################################################")


        if self.relaxed == True:
            print("########################################################")
            print("                       NOTE                             ")
            print(" You are setting relaxed flag to (True), You have to    ")
            print(" provide relaxed path & fdf name for both (endpoint)    ")
            print(" initial and final structures!                          ")
            print("########################################################")

            print ("=================================================")
            print ("The Relaxed Interstitial Image Generation ...")
            print ("=================================================")
            if not self.initial_relaxed_path.exists()  or not self.final_relaxed_path.exists()  :
                raise FileNotFoundError ("intial/final relaxed path not provided")
            if not (self.initial_relaxed_path/self.initial_relaxed_fdf_name).exists()  or not (self.final_relaxed_path/self.final_relaxed_fdf_name).exists() :
                raise FileNotFoundError("intial/final relaxed fdf not provided")
            self.initial_structure = sisl.get_sile(self.initial_relaxed_path/ self.initial_relaxed_fdf_name).read_geometry(output=True)
            self.final_structure = sisl.get_sile(self.final_relaxed_path/ self.final_relaxed_fdf_name).read_geometry(output=True)
            self.__info = prepare_ase_for_relaxed(self.initial_structure,self.final_structure,self.ghost)
            self.__initial = sisl.Geometry.toASE(self.initial_structure)
            self.__final = sisl.Geometry.toASE(self.final_structure)
 
        
        else:
             print ("=================================================")
             print ("The Initial Interstitial Image Generation ...")
             print ("=================================================")
             self.__i_index = finding_atom_position_index(self.pristine_structure , self.initial_atom_position)
             #self.__f_index = finding_atom_position_index (self.pristine_structure , self.final_vacancy_position)
             self.__info = prepare_ase_for_interstitial(self.pristine_structure, self.__i_index , self.final_atom_position,self.ghost)
             self.__initial  = self.__info['initial']
             self.__final  = self.__info['final']

    def prepare_interstitial_images(self):
        """
        """
        
        from ase.neb import NEB
        import sisl

        images_ASE = [self.__initial]
        print ("Copying ASE For NEB Image  0 (initial)")
        for i in range(self.number_of_images):
            print ("Copying ASE For NEB Image ",i+1)
            images_ASE.append(self.__initial.copy())
        images_ASE.append(self.__final)
        print ("Copying ASE For NEB Image ",i+2, 'final')
        self.neb = NEB(images_ASE)
        self.neb.interpolate(self.interpolation_method,mic=True)
       
        self.images = []
        for i in range(self.number_of_images+2):
            self.images.append(sisl.Geometry.fromASE(images_ASE[i]))

        if self.ghost : #== True and self.relaxed is not True:
            for i in range(self.number_of_images+2):
                print(f" Adding Ghost for image {i} in Sisl Geometry Object for initial ")
                self.images[i] = self.images[i].add(self.__info['ghost_initial'])
                print(f" Adding Ghost for image {i} in Sisl Geometry Object for final ")
                self.images[i] = self.images[i].add(self.__info['ghost_final'])

        # This is may be! a new feature, i am still testing it though!
        if self.moving == True and self.relaxed is not True:
            for i in range(self.number_of_images+2):
                if self.ghost:
                    self.images[i].atom[-3] = sisl.Atom(Z = self.images[i].atom[-3].Z,tag=self.images[i].atoms[-3].symbol+"_moving")
                else:
                    self.images[i].atom[-1] = sisl.Atom(Z = self.images[i].atom[-1].Z,tag=self.images[i].atoms[-1].symbol+"_moving")

        if self.relaxed is not True:
            self.fdf_image = self.images[0]
            self._initial = self.images[0]
            self._final = self.images[-1]
    
        if self.relaxed :
            self.fdf_image = self.initial_structure

class KickNEB(SiestaBarriersBaseNick):
    """
    """
    def __init__(self,
            pristine_structure,
            initial_atom_position,
            final_atom_position,
            kicked_atom_final_position,
            number_of_images,
            interpolation_method = 'idpp',
            initial_relaxed_path = '',
            final_relaxed_path = '',
            initial_relaxed_fdf_name = 'input.fdf',
            final_relaxed_fdf_name = 'input.fdf',
            ghost = True,
            moving = False,
            relaxed = False,
            neb_path = 'neb',
            path = 'image' ):

        super().__init__(neb_path = neb_path,
                         path='image',
                         images = [pristine_structure],
                         initial_relaxed_path = pathlib.Path(initial_relaxed_path),
                         final_relaxed_path = pathlib.Path(final_relaxed_path),
                         initial_relaxed_fdf_name = initial_relaxed_fdf_name,
                         final_relaxed_fdf_name = final_relaxed_fdf_name
                         )

        self.pristine_structure = pristine_structure
        self.initial_atom_position = initial_atom_position
        self.final_atom_position = final_atom_position
        self.kicked_atom_final_position = kicked_atom_final_position
        self.number_of_images = number_of_images
        self.interpolation_method = interpolation_method
        self.initial_relaxed_fdf_name = initial_relaxed_fdf_name
        self.final_relaxed_fdf_name = final_relaxed_fdf_name
        self.ghost = ghost
        self.moving = moving
        self.relaxed = relaxed
        
        if self.relaxed == True:
            print ("=================================================")
            print ("The Relaxed Kicked Image Generation ...")
            print ("=================================================")
            if not self.initial_relaxed_path.exists()  or not self.final_relaxed_path.exists()  :
                raise FileNotFoundError ("intial/final relaxed path not provided")
            if not (self.initial_relaxed_path/self.initial_relaxed_fdf_name).exists()  or not (self.final_relaxed_path/self.final_relaxed_fdf_name).exists() :
                raise FileNotFoundError("intial/final relaxed fdf not provided")
            self.initial_structure = sisl.get_sile(self.initial_relaxed_path/ self.initial_relaxed_fdf_name).read_geometry(output=True)
            self.final_structure = sisl.get_sile(self.final_relaxed_path/ self.final_relaxed_fdf_name).read_geometry(output=True)
            self.__initial = sisl.Geometry.toASE(self.initial_structure)
            self.__final = sisl.Geometry.toASE(self.final_structure)
 
        
        else:
             print ("=================================================")
             print ("The Initial Kicked Image Generation ...")
             print ("=================================================")
             self.__i_index = finding_atom_position_index(self.pristine_structure , self.initial_atom_position)
             self.__f_index = finding_atom_position_index (self.pristine_structure , self.final_atom_position)
             self.__info = prepare_ase_for_kick(self.pristine_structure, self.__i_index , self.__f_index,self.kicked_atom_final_position,self.ghost)
            # self.__initial  = self.__info['initial']
            # self.__final  = self.__info['final']
             self.__initial  = self.__info['initial']
             self.__final  = self.__info['final']


    def prepare_kick_images(self):
        """
        """
        
        from ase.neb import NEB
        import sisl

        images_ASE = [self.__initial]
        print ("Copying ASE For NEB Image  0 (initial)")
        for i in range(self.number_of_images):
            print ("Copying ASE For NEB Image ",i+1)
            images_ASE.append(self.__initial.copy())
        images_ASE.append(self.__final)
        print ("Copying ASE For NEB Image ",i+2, 'final')
        self.neb = NEB(images_ASE)
        self.neb.interpolate(self.interpolation_method,mic=True)
       
        #-------------------------------------------------------------------
        # For Kick
        #-------------------------------------------------------------------
        if self.relaxed == True:
            d = self.final_atom_position - self.initial_atom_position
        else:
            d = self.final_atom_position - self.initial_atom_position
        Steps = d / (self.number_of_images +1)

        if self.relaxed == True:
            FinalAtomPositionKick = self.__info['trace_atom_B_initial'].xyz[0]
        else:
            FinalAtomPositionKick = self.final_atom_position
        MovingAtomIndex=len(self.neb.images[0].get_positions())
        MovingAtomKick=np.array([])
        for l in range(self.neb.nimages):
            if l==0:
                MovingAtomKick=np.append(MovingAtomKick,FinalAtomPositionKick)
            if l>0:
                MovingAtomKick=np.append(MovingAtomKick,FinalAtomPositionKick+Steps)
                FinalAtomPositionKick=FinalAtomPositionKick+Steps
        MovingAtomKick=MovingAtomKick.reshape(self.number_of_images+2,3)

        if self.relaxed == True:
            steps_x = np.divide(self.__info['trace_atom_B_kicked'].xyz[0][0]-MovingAtomKick[0][0],  len(MovingAtomKick))
            steps_y = np.divide(self.__info['trace_atom_B_kicked'].xyz[0][1]-MovingAtomKick[0][1],  len(MovingAtomKick))
            steps_z = np.divide(self.__info['trace_atom_B_kicked'].xyz[0][2]-MovingAtomKick[0][2],  len(MovingAtomKick))

        else :
            steps_x = np.divide(self.kicked_atom_final_position[0]-MovingAtomKick[0][0],  len(MovingAtomKick))
            steps_y = np.divide(self.kicked_atom_final_position[1]-MovingAtomKick[0][1],  len(MovingAtomKick))
            steps_z = np.divide(self.kicked_atom_final_position[2]-MovingAtomKick[0][2],  len(MovingAtomKick))
        print (steps_x)
        print (steps_y)
        print (steps_z)
        #Offset
        Offset = np.array([])
        for l in range(len(MovingAtomKick)):
            if l == 0:
                Offset=np.append(Offset,0.0)
                Offset=np.append(Offset,0.0)
                Offset=np.append(Offset,0.0)
            else:
                Offset=np.append(Offset,steps_x*l + steps_x)
                Offset=np.append(Offset,steps_y*l + steps_y)
                Offset=np.append(Offset,steps_z*l + steps_z)
        Offset=Offset.reshape(len(MovingAtomKick),3)

        MovingAtomKick=Offset+MovingAtomKick[0]
        self.MovingAtomKick = MovingAtomKick
        print("DEBUG: {}".format(self.MovingAtomKick))
        sisl_moving=[]

       # Fixing the Tag
        self.KickedAtomInfo = self.__info['trace_atom_B_kicked']
        print("DEBUG: {}".format(self.KickedAtomInfo))
        for i in range(self.number_of_images+2):
            sisl_moving.append(sisl.Geometry(xyz = self.MovingAtomKick[i],
                                             atoms = sisl.Atom(Z = self.KickedAtomInfo.atom[0].Z,tag=self.KickedAtomInfo.atoms.atom[0].symbol+"_kicked")))


        self.images = []
        for i in range(self.number_of_images+2):
            self.images.append(sisl.Geometry.fromASE(images_ASE[i]))
            self.images[i] = self.images[i].add(sisl_moving[i])

        

        if self.ghost == True and self.relaxed is not True:
            for i in range(self.number_of_images+2):
                print(f" Adding Ghost for image {i} in Sisl Geometry Object for initial ")
                self.images[i] = self.images[i].add(self.__info['ghost_initial'])
                print(f" Adding Ghost for image {i} in Sisl Geometry Object for final ")
                self.images[i] = self.images[i].add(self.__info['ghost_final'])

        ## This is may be! a new feature, i am still testing it though!
        if self.moving == True and self.relaxed is not True:
            for i in range(self.number_of_images+2):
                if self.ghost:
                    self.images[i].atom[-4] = sisl.Atom(Z = self.images[i].atom[-4].Z,tag=self.images[i].atoms[-4].symbol+"_moving")
                else:
                    self.images[i].atom[-2] = sisl.Atom(Z = self.images[i].atom[-2].Z,tag=self.images[i].atoms[-2].symbol+"_moving")

        if self.relaxed is not True:
            self.fdf_image = self.images[0]
            self._initial = self.images[0]
            self._final = self.images[-1]
    
        if self.relaxed :
            self.fdf_image = self.initial_structure


