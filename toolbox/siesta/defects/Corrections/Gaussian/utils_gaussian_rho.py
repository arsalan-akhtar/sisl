import numpy as np

def get_shift_initial_rho_model(charge_grid,shift_grid,geometry):
    """
    chrage_grid : sisl grid
    shift_grid : grid points where to shift
    """
    import sisl
    import numpy as np
    from scipy.ndimage.interpolation import shift

    result_shifted = shift(charge_grid,shift=shift_grid,mode='grid-wrap')
    result_grid_shifted  = sisl.Grid(result_shifted.shape)
    result_grid_shifted.grid = result_shifted
    result_grid_shifted.set_geometry(geometry)
    return result_grid_shifted 


def get_rho_model(charge_grid,geometry,scale_f,sub_grid_shift,write_out=False):
    """
    """
    import sisl
    import numpy as np
    super_cell = geometry.tile(scale_f,0).tile(scale_f,1).tile(scale_f,2)
    shape_sc = ( scale_f* charge_grid.shape[0],
                 scale_f* charge_grid.shape[1],
                 scale_f* charge_grid.shape[2])
    print(f"DEBUG: shape of scale model {shape_sc}")
    grid_SC = sisl.Grid( shape = shape_sc ,
                        geometry = super_cell)
    print(f"DEBUG: sub grid shift {sub_grid_shift}")
    #grid_SC.grid = np.pad(charge_grid.grid,sub_grid_shift,mode='empty')
    grid_SC.grid = np.pad(charge_grid.grid,sub_grid_shift,mode='constant',constant_values=(0, 0))

    if write_out:
        grid_SC.write(f"model_charge-{scale_f}{scale_f}{scale_f}.XSF")
    return grid_SC


def shift_prepare(defect_site,grid):
    """
    """
    if defect_site[0]!=0.0:
        n_x = int(defect_site[0]/grid.dcell[0][0])+1
    else:
        n_x = 0
    if defect_site[1]!=0.0:
        n_y = int(defect_site[1]/grid.dcell[1][1])+1
    else:
        n_y = 0
    if defect_site[2]!=0.0:
        n_z = int(defect_site[2]/grid.dcell[2][2])+1
    else:
        n_z = 0

    
    print (f" Mesh grid {grid.shape}")
    print (f" Defect site position on mesh {n_x,n_y,n_z}")
    if n_x >= grid.shape[0]/2:
        d_position_x = n_x*grid.dcell[0][0]-grid.dcell[0][0]
    else:
        d_position_x = n_x*grid.dcell[0][0]
        
    if n_y >= grid.shape[1]/2:
        d_position_y = n_y*grid.dcell[1][1]-grid.dcell[1][1]
    else:
        d_position_y = n_y*grid.dcell[1][1]

    if n_z >= grid.shape[2]/2:
        d_position_z = n_z*grid.dcell[2][2]-grid.dcell[2][2]
    else:
        d_position_z = n_z*grid.dcell[2][2]
        
    d_position = (d_position_x,
                  d_position_y,
                  d_position_z)
    print (f" Defect site position on crystal {d_position}")
    #print
    #===========
    if n_x<0:
        print("y is negative")
        #n_y_new = n_y
        #n_y_new = grid.shape[1]+int(defect_site[1]/grid.dcell[1][1])#-1
        print(n_x)
        n_x=abs(n_x)

    if n_y<0:
        print("y is negative")
        #n_y_new = n_y
        #n_y_new = grid.shape[1]+int(defect_site[1]/grid.dcell[1][1])#-1
        print(n_y)
        n_y=abs(n_y)
        
    if n_z<0:
        print("z is negative")
        #n_y_new = n_y
        #n_y_new = grid.shape[1]+int(defect_site[1]/grid.dcell[1][1])#-1
        print(n_z)
        n_z=abs(n_z)
        
    if n_x >= grid.shape[0]/2:
        s_x = n_x - int(grid.shape[0]/2) #0 #int(grid.shape[0]/2)+int((n_x-int(grid.shape[0]/2)))
    else:
        s_x = int((int(grid.shape[0]/2)-n_x))
    if n_y >= grid.shape[1]/2:
        s_y = n_y - int(grid.shape[1]/2) #0 #int(grid.shape[1]/2)+int((n_y-int(grid.shape[1]/2)))
    else:
        s_y = int((int(grid.shape[1]/2)-n_y))
    if n_z >= grid.shape[2]/2:
        s_z = n_z -int(grid.shape[2]/2) #0 #int(grid.shape[2]/2)-int((n_z-int(grid.shape[2]/2)))
    else:
        s_z = int((int(grid.shape[2]/2)-n_z))

    shift=(s_x,s_y,s_z)
    print(f"the Shift is {shift}")
    print(f"the Shift position {s_x*grid.dcell[0][0],s_y*grid.dcell[1][1],s_z*grid.dcell[2][2]}")
    return shift



def shift_prepare_buggy_todelete(defect_site,grid):
    """
    """
    if defect_site[0]!=0.0:
        n_x = int(defect_site[0]/grid.dcell[0][0])+1
    else:
        n_x = 0
    if defect_site[1]!=0.0:
        n_y = int(defect_site[1]/grid.dcell[1][1])+1
    else:
        n_y = 0
    if defect_site[2]!=0.0:
        n_z = int(defect_site[2]/grid.dcell[2][2])+1
    else:
        n_z = 0
    print (f" Defect site position on mesh {n_x,n_y,n_z}")
    print (f" Defect site recheck {n_x*grid.dcell[0][0],n_y*grid.dcell[1][1],n_z*grid.dcell[2][2]}")
    
    if n_x >= grid.shape[0]/2:
        s_x = int(grid.shape[0]/2)+int((n_x-int(grid.shape[0]/2)))
    else:
        s_x = int((int(grid.shape[0]/2)-n_x))
    if n_y >= grid.shape[0]/2:
        s_y = int(grid.shape[0]/2)+int((n_y-int(grid.shape[0]/2)))
    else:
        s_y = int((int(grid.shape[0]/2)-n_y))  
    if n_z >= grid.shape[0]/2:
        s_z = int(grid.shape[0]/2)+int((n_z-int(grid.shape[0]/2)))
    else:       
        s_z = int((int(grid.shape[0]/2)-n_z))

    shift=(s_x,s_y,s_z)
    return shift

def reverse_shift_prepare(defect_site,grid):
    """
    Only for rho bcz i put it in center of cell
    this will bring back to the point of defects
    """
    if defect_site[0]!=0.0:
        n_x = int(defect_site[0]/grid.dcell[0][0])+1
    else:
        n_x = 0
    if defect_site[1]!=0.0:
        n_y = int(defect_site[1]/grid.dcell[1][1])+1
    else:
        n_y = 0
    if defect_site[2]!=0.0:
        n_z = int(defect_site[2]/grid.dcell[2][2])+1
    else:
        n_z = 0
    print (f" Defect site position on mesh {n_x,n_y,n_z}")
    print (f" Defect site recheck {n_x*grid.dcell[0][0],n_y*grid.dcell[1][1],n_z*grid.dcell[2][2]}")

    if n_x >= grid.shape[0]/2:
        s_x = int(grid.shape[0]/2)+int((n_x-int(grid.shape[0]/2)))
    else:
        s_x = int((int(grid.shape[0]/2)-n_x))
    if n_y >= grid.shape[0]/2:
        s_y = int(grid.shape[0]/2)+int((n_y-int(grid.shape[0]/2)))
    else:
        s_y = int((int(grid.shape[0]/2)-n_y))
    if n_z >= grid.shape[0]/2:
        s_z = int(grid.shape[0]/2)+int((n_z-int(grid.shape[0]/2)))
    else:
        s_z = int((int(grid.shape[0]/2)-n_z))

    shift=(-1*s_x,-1*s_y,-1*s_z)
    return shift


def cut_rho(charge_density,tolerance,struct):
    """
    """
    import sisl
    #mask = np.ma.greater(np.abs(charge_density.grid), tolerance)
    mask = np.ma.greater(charge_density.grid, tolerance)
    charge_density_cutted = np.ma.masked_array(charge_density.grid , mask=mask)
    
    rho = sisl.Grid(charge_density_cutted.shape , geometry = struct)
    rho.grid = charge_density_cutted
    return rho


def get_rotate_grid_sisl(grid,angle,axes,geometry,reshape=False,prefilter=True,mode="grid-wrap",f_name="V_rotate",save=False):
    """
    """
    from scipy import ndimage, misc
    import sisl
    #img = misc.ascent()
    rot_arr=ndimage.rotate(grid.grid, angle=angle,axes=axes, reshape=reshape,mode=mode,prefilter=prefilter)
    grid_rotate = sisl.Grid(rot_arr.shape)
    grid_rotate.grid =rot_arr
    grid_rotate.set_geometry(geometry)
    filename = f"{f_name}.XSF"
    if save:
        grid_rotate.write(filename)
    return grid_rotate
