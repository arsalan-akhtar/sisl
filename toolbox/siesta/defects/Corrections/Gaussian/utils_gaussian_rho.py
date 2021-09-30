

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



