class DataManager:
    def __init__(self):
        self.param_to_var_mapping = {}
        self.stored_values = {}
        

    def __init__(self):

        self.param_to_var_mapping = {
            'X_min': 'x_min',
            'X_max': 'x_max',
            'Y_min': 'y_min',
            'Y_max': 'y_max',
            'Z_min': 'z_min',
            'Z_max': 'z_max',
            'Magnitude_min': 'magnitude_min',
            'Magnitude_max': 'magnitude_max',
            'Moment_min': 'moment_min',
            'Moment_max': 'moment_max',
            'Marker Range_min': 'marker_range_min',
            'Marker Range_max': 'marker_range_max',
            'R_min': 'r_min',
            'R_max': 'r_max',
            'N_value': 'n_value',
            'bins_value': 'bins_value',
            'Alpha_value': 'alpha_value',
            'Grid Spacing_value': 'grid_spacing',

            'Cut step_value': 'cut_step',
            'Xm_value': 'xm_value',
            'Ym_value': 'ym_value',
            'Zm_value': 'zm_value',
            'Radius_value': 'radius_value',
        }
        self.stored_values = {}

    def update_values(self, parameter_rows):
        try:
            self.stored_values = {}
            for parameter_name, widget in parameter_rows.items():
                if hasattr(widget, 'inputs'):  
                    for identifier, input_widget in widget.inputs.items():
                        key = f"{parameter_name}_{identifier}"
                        
                        if parameter_name in ['Time', 'Date']:
                            self.stored_values[key] = input_widget.text
                        else:
                            try:
                                self.stored_values[key] = float(input_widget.text)
                            except ValueError:
                                self.stored_values[key] = input_widget.text
                else:
                    if hasattr(widget, 'text'):
                        self.stored_values[parameter_name] = widget.text
        except Exception as e:
            print(f"Error in processing input values: {e}")

    def get_value(self, key):
        return self.stored_values.get(key)

    def get_mapped_value(self, var_name):
        key = self.param_to_var_mapping.get(var_name)
        return self.get_value(key) if key else None