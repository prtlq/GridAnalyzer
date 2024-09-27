import pandas as pd
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta


from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.utils import get_color_from_hex
from kivy.clock import Clock

from data_handling import select_csv_file, read_csv, DataProcessor
from configmanifest import DataManager


# Constants for UI
LABEL_WIDTH = 100
SLIDER_HEIGHT = '48dp'
EXECUTE_BUTTON_COLOR = get_color_from_hex('#5AC6ED')
BACKGROUND_COLOR_HEX = '#949494'
TEXT_COLOR_HEX = "#212121"
SECTION_TITLE_COLOR_HEX = "#1B596F"

logging.basicConfig(level=logging.DEBUG)



class ParameterRow(BoxLayout):
    def __init__(self, parameter_name, has_inputs=True, number_of_inputs=2, default_values=None, **kwargs):
        super().__init__(orientation='horizontal', size_hint_y=None, height=30, **kwargs)
        self.inputs = {}
        self.add_widget(Label(text=parameter_name, size_hint_x=None, width=LABEL_WIDTH, halign='left',
                              font_size=16, color=get_color_from_hex(TEXT_COLOR_HEX)))
        if has_inputs:
            identifiers = ['min', 'max'] if number_of_inputs == 2 else ['value']
            for i in range(number_of_inputs):
                identifier = identifiers[i]
                default_value = default_values[i] if default_values and i < len(default_values) else ''
                text_input = TextInput(text=str(default_value), size_hint_x=0.4, width=LABEL_WIDTH, multiline=False)
                text_input.bind(text=self.on_text_change)
                self.inputs[identifier] = text_input
                self.add_widget(text_input)

    def on_text_change(self, instance, value):
        identifier = next((key for key, widget in self.inputs.items() if widget == instance), None)
        #print(f"New value in '{instance.parent.children[-1].text} {identifier}': {value}")

class DateTimeRangeSelector(BoxLayout):
    def __init__(self, min_datetime, max_datetime, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.min_datetime = min_datetime
        self.max_datetime = max_datetime

        self.start_datetime = min_datetime
        self.end_datetime = max_datetime


        self.start_slider = Slider(min=self.min_datetime.timestamp(),
                                   max=self.max_datetime.timestamp(),
                                   value=self.min_datetime.timestamp())
        self.end_slider = Slider(min=self.min_datetime.timestamp(),
                                 max=self.max_datetime.timestamp(),
                                 value=self.max_datetime.timestamp())


        self.start_datetime_input = TextInput(text=self.start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                                              multiline=False,
                                              size_hint_y=None,
                                              height=30,
                                              halign='center')


        self.end_datetime_input = TextInput(text=self.end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                                              multiline=False,
                                              size_hint_y=None,
                                              height=30,
                                              halign='center')

        self.start_slider.bind(value=self.on_slider_value_change)
        self.end_slider.bind(value=self.on_slider_value_change)
        self.start_datetime_input.bind(text=self.on_datetime_input_change)
        self.end_datetime_input.bind(text=self.on_datetime_input_change)

        self.add_widget(self.start_datetime_input)
        self.add_widget(self.start_slider)
        self.add_widget(self.end_slider)
        self.add_widget(self.end_datetime_input)

    def on_slider_value_change(self, instance, value):
        new_datetime = pd.to_datetime(value, unit='s')
        if instance == self.start_slider:
            self.start_datetime = new_datetime
            self.start_datetime_input.text = self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')
        elif instance == self.end_slider:
            self.end_datetime = new_datetime
            self.end_datetime_input.text = self.end_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    def on_datetime_input_change(self, instance, value):
        try:
            new_datetime = pd.to_datetime(value)
            if instance == self.start_datetime_input:
                if new_datetime > self.end_datetime:
                    instance.text = self.end_datetime.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    self.start_datetime = new_datetime
                    self.start_slider.value = self.start_datetime.timestamp()
            elif instance == self.end_datetime_input:
                if new_datetime < self.start_datetime:
                    instance.text = self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    self.end_datetime = new_datetime
                    self.end_slider.value = self.end_datetime.timestamp()
        except ValueError:
            # self.add_message("Date/Time format is not valid.")
            pass

    def update_range(self, min_datetime, max_datetime):
        self.min_datetime = min_datetime
        self.max_datetime = max_datetime

        self.start_slider.min = self.min_datetime.timestamp()
        self.start_slider.max = self.max_datetime.timestamp()
        self.start_slider.value = self.min_datetime.timestamp()

        self.end_slider.min = self.min_datetime.timestamp()
        self.end_slider.max = self.max_datetime.timestamp()
        self.end_slider.value = self.max_datetime.timestamp()

        # self.start_label.text = f"Start: {self.min_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        # self.end_label.text = f"End: {self.max_datetime.strftime('%Y-%m-%d %H:%M:%S')}"

class GridAnalyzer(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_rows = {}
        self.data_manager = DataManager()
        self.processor = DataProcessor()
        self.selected_axis = None
        self.selected_column = None
        self.cut_step = None
        self.selected_axis = "X"
        self.selected_view = None
        self.scrollable_layout = GridLayout(cols=1, size_hint_y=None, spacing=1, padding=1)

    def on_start(self):
        self.root.bind(width=self.adjust_middle_panel_width)
        #App.get_running_app().root_window.fullscreen = 'auto'

    def build(self):
        main_layout = self.setup_main_layout()
        return main_layout
    
    def setup_background(self, layout):
        with layout.canvas.before:
            Color(*get_color_from_hex('#949494'))
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
        layout.bind(pos=self.update_rect, size=self.update_rect)

    def setup_main_layout(self):
        main_layout = BoxLayout(orientation='horizontal', spacing=0)
        self.setup_background(main_layout)

        left_panel = self.setup_left_panel()
        left_panel.size_hint_x = 0.2

        middle_panel = self.setup_middle_panel()
        middle_panel.size_hint_x = 0.6

        right_panel = self.setup_right_panel()
        right_panel.size_hint_x = 0.2

        main_layout.add_widget(left_panel)
        main_layout.add_widget(middle_panel)
        main_layout.add_widget(right_panel)

        return main_layout
    
    def create_button_container(self):
        container = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        container.add_widget(BoxLayout())
        container.add_widget(BoxLayout())  
        return container

    def create_spacer(self, height):
        return BoxLayout(size_hint_y=None, height=height)

    def adjust_middle_panel_width(self, instance, new_width):
        if hasattr(self, 'middle_panel'):
            self.middle_panel.width = new_width * 0.8

    def setup_left_panel(self):
        left_panel = BoxLayout(orientation='vertical', size_hint=(0.2, 1))

        file_select_button_container = BoxLayout(size_hint=(1, None), height=50)
        file_select_button = Button(text='Select the File', size_hint=(1, None), height=50)
        file_select_button.bind(on_press=self.on_file_select_button_click)
        file_select_button_container.add_widget(file_select_button)
        left_panel.add_widget(file_select_button_container)

        self.scrollable_layout = GridLayout(cols=1, size_hint_y=None, spacing=1, padding=1)
        self.scrollable_layout.bind(minimum_height=self.scrollable_layout.setter('height'))

        parameters = [
            ("Initial Range", False),  ("Location", False), ("X", True), ("Y", True),
            ("Z", True), ("Parameter", False), ("Magnitude", True), ("Moment", True)
        ]

        for param, has_input in parameters:
            row = ParameterRow(param, has_inputs=has_input)
            self.parameter_rows[param] = row
            self.scrollable_layout.add_widget(row)

        self.scrollable_layout.add_widget(self.create_spacer(10))
        self.marker_spinner = Spinner(text='Marker', values=(), size_hint=(1, None), height=40)
        self.marker_spinner.bind(text=self.marker_spinner_select)
        self.scrollable_layout.add_widget(self.marker_spinner)

        self.scrollable_layout.add_widget(self.create_spacer(10))

        self.parameter_rows['Marker Range'] = ParameterRow('Marker Range', has_inputs=True)
        self.scrollable_layout.add_widget(self.parameter_rows['Marker Range'])

        self.scrollable_layout.add_widget(self.create_spacer(30))

        default_r_values = ['2', '8']
        r_row = ParameterRow('R', has_inputs=True, number_of_inputs=2, default_values=default_r_values)
        self.parameter_rows['R'] = r_row
        self.scrollable_layout.add_widget(r_row)

        default_n_value = ['50']
        n_row = ParameterRow('N', has_inputs=True, number_of_inputs=1, default_values=default_n_value)
        self.parameter_rows['N'] = n_row
        self.scrollable_layout.add_widget(n_row)

        self.scrollable_layout.add_widget(self.create_spacer(30))

        execute_button = Button(
            text='Execute',
            background_color=get_color_from_hex('#5AC6ED'),
            size_hint=(1, None),
            height=50 
        )
        execute_button.bind(on_press=self.on_execute_button_click)

        self.scrollable_layout.add_widget(execute_button)
        self.scrollable_layout.add_widget(self.create_spacer(80))

        slider_container = BoxLayout(orientation='vertical', size_hint=(1, None), height=150)
        default_min_datetime = pd.to_datetime("2000-01-01")
        default_max_datetime = pd.to_datetime("2030-12-31")
        self.datetime_range_selector = DateTimeRangeSelector(default_min_datetime, default_max_datetime)
        slider_container.add_widget(self.datetime_range_selector)
        left_panel.add_widget(slider_container)
        
        scroll_view = ScrollView(size_hint=(1, 1), do_scroll_x=False, do_scroll_y=True)
        scroll_view.add_widget(self.scrollable_layout)
        left_panel.add_widget(scroll_view)

        self.message_box = TextInput(multiline=True, readonly=True, font_size='12sp', hint_text='This software is the proprietary asset of AFRY, developed as a delivery to LKAB. Modification or distribution should be permitted by the authors. Developed by Pouria Taleghani, Hamid Sabeti, Anneliese Botelho', size_hint_y=None, height=150)
        left_panel.add_widget(self.message_box)


        return left_panel

    def setup_right_panel(self):
        right_panel = BoxLayout(orientation='vertical', size_hint=(1, 1))
        scrollable_layout = GridLayout(cols=1, size_hint_y=None, spacing=1, padding=1)
        scrollable_layout.bind(minimum_height=scrollable_layout.setter('height'))

        sections = [
            ('Histogram', [('bins', True, 1, ['20']), ('Alpha', True, 1, ['0.7'])]),
            ('Block', [('Grid Spacing', True, 1, ['20']), ]),
            ('Plane', [('Cut step', True, 1, ['50']), ('Axis', 'axis_spinner', ['X', 'Y', 'Z']), ('Min cut', True, 1, ['1000']), ('Max cut', True, 1, ['1200'])]),
            # ('Event Marker', [('Xm', True, 1, ['0']), ('Ym', True, 1, ['0']), ('Zm', True, 1, ['0']), ('Radius', True, 1, ['10'])]),
            ('Batch Process', [
                ('Event time', True, 1, [self.end_datetime_input.text if hasattr(self, 'end_datetime_input') else '']), 
                ('Backtrack (d)', True, 1, ['7']),  
                ('Interval (d)', True, 1, ['1'])  
            ])
        ]
        def create_section_title(text):
            section_title = Label(text=text, size_hint=(None, None), size=(100, 30),
                                font_size=16, color=get_color_from_hex("#1B596F"))
            return section_title

        for section_title, rows in sections:
            section_layout = BoxLayout(orientation='horizontal', size_hint=(None, None), height=50)
            section_title_widget = create_section_title(section_title)
            section_layout.add_widget(section_title_widget)
            scrollable_layout.add_widget(section_layout)

            for row in rows:
                if row[1] == 'axis_spinner':
                    spinner_title = row[0]
                    spinner_options = row[2]

                    spinner_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
                    spinner_label = Label(text=spinner_title, size_hint_x=None, width=100, font_size=16, color=get_color_from_hex("#212121"))
                    
                    self.axis_spinner = Spinner(
                        text=spinner_options[0],
                        values=spinner_options,
                        size_hint_x=1,
                        font_size=16)
                    
                    self.axis_spinner.bind(text=self.axis_spinner_select)

                    spinner_layout.add_widget(spinner_label)
                    spinner_layout.add_widget(self.axis_spinner)

                    self.parameter_rows[f"{spinner_title}_spinner"] = self.axis_spinner
                    scrollable_layout.add_widget(spinner_layout)

                elif row[0] == 'Grid Spacing':
                    parameter_row = ParameterRow(row[0], row[1], row[2], row[3])
                    self.parameter_rows[row[0]] = parameter_row
                    scrollable_layout.add_widget(parameter_row)

                    view_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)

                    view_label = Label(text="View", size_hint_x=None, width=100, color=get_color_from_hex("#212121"))
                    view_layout.add_widget(view_label)

                    radio_button_xy = ToggleButton(text='XY', group='view', size_hint_x=1)
                    radio_button_xy.bind(on_press=self.update_view)
                    radio_button_yz = ToggleButton(text='YZ', group='view', size_hint_x=1)
                    radio_button_yz.bind(on_press=self.update_view)
                    radio_button_xz = ToggleButton(text='XZ', group='view', size_hint_x=1)
                    radio_button_xz.bind(on_press=self.update_view)

                    view_layout.add_widget(radio_button_xy)
                    view_layout.add_widget(radio_button_yz)
                    view_layout.add_widget(radio_button_xz)

                    scrollable_layout.add_widget(view_layout)

                elif row[0] == 'Interval (d)':
                    parameter_row = ParameterRow(row[0], row[1], row[2], row[3])
                    self.parameter_rows[row[0]] = parameter_row
                    scrollable_layout.add_widget(parameter_row)

                    batch_opt_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
                    batch_opt_label = Label(text="Batch Opt", size_hint_x=None, width=100, color=get_color_from_hex("#212121"))
                    batch_opt_checkbox = CheckBox(active=False, size_hint_x=None, width=30)

                    self.batch_opt_checkbox = CheckBox(active=False, size_hint_x=None, width=30)
                    batch_opt_layout.add_widget(batch_opt_label)
                    batch_opt_layout.add_widget(self.batch_opt_checkbox)
                    scrollable_layout.add_widget(batch_opt_layout)

                else:
                    row_title, has_inputs, number_of_inputs, default_values = row
                    parameter_row = ParameterRow(row_title, has_inputs, number_of_inputs, default_values)
                    self.parameter_rows[row_title] = parameter_row
                    scrollable_layout.add_widget(parameter_row)

        event_time_row = self.parameter_rows['Event time']
        if hasattr(self, 'end_datetime_input'):
            self.end_datetime_input.bind(text=lambda instance, value: setattr(event_time_row.inputs['value'], 'text', value))
                
        scroll_view = ScrollView(size_hint=(1, 1), do_scroll_x=False, do_scroll_y=True)
        scroll_view.add_widget(scrollable_layout)
        right_panel.add_widget(scroll_view)

        return right_panel

    def setup_middle_panel(self):
        middle_panel = BoxLayout(orientation='vertical', size_hint=(0.6, 1))
        self.plot_tabs = TabbedPanel(do_default_tab=False, size_hint=(1, 1))

        tab_names = ['Histogram', 'Block', 'Plane']
        self.plot_areas = {}

        for i, name in enumerate(tab_names):
            tab = TabbedPanelItem(text=name)
            grid_layout = GridLayout(cols=1)

            plot_area = BoxLayout()
            self.plot_areas[name] = plot_area
            grid_layout.add_widget(plot_area)

            tab.add_widget(grid_layout)
            self.plot_tabs.add_widget(tab)

        middle_panel.add_widget(self.plot_tabs)

        return middle_panel

    def marker_spinner_select(self, spinner, text):

        self.selected_column = text
        if hasattr(self, 'data') and text in self.data.columns:
            column_data = self.data[text].dropna()
            min_value, max_value = column_data.min(), column_data.max()

            self.update_parameter_row('Marker Range', [min_value, max_value])

    def axis_spinner_select(self, spinner, text):

        self.selected_axis = text
        print(f"Selected Axis for plane cut: {self.selected_axis}")

        axis_index_mapping = {'X': 0, 'Y': 1, 'Z': 2}
        if self.selected_axis in axis_index_mapping and self.processor.df is not None:
            column_index = axis_index_mapping[self.selected_axis]
            cut_min_value = self.processor.df.iloc[:, column_index].min()
            cut_max_value = self.processor.df.iloc[:, column_index].max()

            self.parameter_rows['Min cut'].inputs['value'].text = str(cut_min_value)
            self.parameter_rows['Max cut'].inputs['value'].text = str(cut_max_value)

            print(f"Updated Min cut: {cut_min_value}, Max cut: {cut_max_value} based on selected axis")
        else:
            print("Invalid axis selected or DataFrame not available.")

    def get_middle_panel(self):
        root = self.root
        if len(root.children) > 1:
            return root.children[1]
        else:
            return None

    def get_current_tab_text(self):
        if self.plot_tabs:
            return self.plot_tabs.current_tab.text
        else:
            return None

    def on_file_select_button_click(self, instance):
        file_path = select_csv_file()
        if file_path:
            data = read_csv(file_path)
            self.data = data

            if data is not None:
                message = f"Read data from: {file_path}"

                if 'DateTime' in data.columns:
                    min_datetime = data['DateTime'].min()
                    max_datetime = data['DateTime'].max()

                    self.update_parameter_row('Date', [min_datetime.strftime('%Y-%m-%d'), max_datetime.strftime('%Y-%m-%d')])
                    self.update_parameter_row('Time', [min_datetime.strftime('%H:%M:%S'), max_datetime.strftime('%H:%M:%S')])


                    self.min_datetime = min_datetime
                    self.max_datetime = max_datetime
                    self.update_datetime_range_selector(min_datetime, max_datetime)
                self.update_marker_options(data.columns.tolist())              

                for i, param in enumerate(['X', 'Y', 'Z', 'Magnitude', 'Moment']):
                    if len(data.columns) > i + 1:
                        column = data[data.columns[i + 1]]
                        min_value, max_value = column.min(), column.max()
                        self.update_parameter_row(param, [min_value, max_value])
            else:
                message = "Failed to read the CSV file."
        else:
            message = "No file selected."
        self.add_message(message)
        
    def on_execute_button_click(self, instance):
        logging.debug("Execute button clicked.")

        if self.batch_opt_checkbox.active:
            self.add_message("Batch operation initiated...")
            event_time_str = self.parameter_rows['Event time'].inputs['value'].text
            back_days = int(self.parameter_rows['Backtrack (d)'].inputs['value'].text)
            interval_days = int(self.parameter_rows['Interval (d)'].inputs['value'].text)

            event_time = datetime.strptime(event_time_str, '%Y-%m-%d %H:%M:%S')
            intervals = [(event_time - timedelta(days=i * interval_days) - timedelta(days=interval_days), event_time - timedelta(days=i * interval_days)) for i in range(back_days // interval_days)]

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(self.execute_batch_operation, start, end) for start, end in intervals]

                for future in futures:
                    future.result()  

            Clock.schedule_once(lambda dt: self.add_message("Batch operation ongoing..."))
        else:
            self.execute_single_operation()

    def execute_single_operation(self):
        if hasattr(self, 'data') and self.selected_column in self.data.columns:
            self.add_message("Starting analytics...")
            logging.debug("Starting analytics thread.")
            threading.Thread(target=self.analytics, daemon=True).start()
        else:
            self.add_message("Marker missed or does not match any operation.")

    def execute_batch_operation(self, start_datetime, end_datetime):
        def update_gui(dt):
            self.datetime_range_selector.start_datetime = start_datetime
            self.datetime_range_selector.end_datetime = end_datetime
            self.datetime_range_selector.start_datetime_input.text = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
            self.datetime_range_selector.end_datetime_input.text = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
            self.add_message(f"for batch from {start_datetime} to {end_datetime}")

            self.execute_single_operation()

        Clock.schedule_once(update_gui)

    def analytics(self):

        self.update_values()
        start_datetime = self.datetime_range_selector.start_datetime
        end_datetime = self.datetime_range_selector.end_datetime

        selected_tab_text = self.get_current_tab_text()

        processor = DataProcessor()  
        
        if selected_tab_text == 'Histogram':
                Clock.schedule_once(lambda dt: self.clear_plot_area(), 0)

                fig = self.processor.histogram_tab ('histogram',self.data, self.data_manager, self.selected_column, start_datetime, end_datetime)
                Clock.schedule_once(lambda dt: self.update_histogram_tab(fig))
                self.add_message("Histogram tab updated.")

        elif selected_tab_text == 'Block':

            average_columns = [self.data.columns[7], self.data.columns[13], self.data.columns[24]]
            cumulative_columns = [self.data.columns[5], self.data.columns[18]]

            if self.selected_column in average_columns:
                Clock.schedule_once(lambda dt: self.clear_plot_area(), 0)                
                fig, df, ax = self.processor.block_tab(
                    self.data,
                    self.data_manager,
                    self.selected_column,
                    start_datetime,
                    end_datetime,
                    interpolation_method='average',
                    update_view=self.update_plot_view, 
                    selected_view=self.selected_view
                )

                self.ax = ax
                self.original_xticks = self.ax.get_xticks()
                self.original_yticks = self.ax.get_yticks()
                self.original_zticks = self.ax.get_zticks()
                Clock.schedule_once(lambda dt: self.update_block_tab(fig))
                self.add_message("Block tab updated.")

            elif self.selected_column in cumulative_columns:
                Clock.schedule_once(lambda dt: self.clear_plot_area(), 0)                
                fig, df, ax = self.processor.block_tab(
                    self.data,
                    self.data_manager,
                    self.selected_column,
                    start_datetime,
                    end_datetime,
                    interpolation_method='cumulative',
                    update_view=self.update_plot_view, 
                    selected_view=self.selected_view
                )

                self.ax = ax
                self.original_xticks = self.ax.get_xticks()
                self.original_yticks = self.ax.get_yticks()
                self.original_zticks = self.ax.get_zticks()
                Clock.schedule_once(lambda dt: self.update_block_tab(fig))
                self.add_message("Block tab updated.")

            else:
                self.add_message("Marker missed or does not match any operation.")

        elif selected_tab_text == 'Plane':
            self.update_plane_tab()

    def clear_plot_area(self, plot_key=None):
        if plot_key:
            plot_area = self.plot_areas.get(plot_key)
            if plot_area:
                plot_area.clear_widgets()
        else:
            for key, plot_area in self.plot_areas.items():
                plot_area.clear_widgets()

    def update_values(self):
        self.data_manager.update_values(self.parameter_rows)

    def update_view(self, instance):
        if instance.state == 'down':
            self.selected_view = instance.text

            if hasattr(self, 'ax'):
                self.ax.set_xticks(self.original_xticks)
                self.ax.set_yticks(self.original_yticks)
                self.ax.set_zticks(self.original_zticks)
                self.ax.xaxis.set_ticklabels([str(int(tick)) for tick in self.original_xticks])
                self.ax.yaxis.set_ticklabels([str(int(tick)) for tick in self.original_yticks])
                self.ax.zaxis.set_ticklabels([str(int(tick)) for tick in self.original_zticks])

                if self.selected_view == 'XY':
                    self.ax.set_zticks([])
                    self.ax.zaxis.set_ticklabels([])
                    self.ax.view_init(elev=90, azim=0)

                elif self.selected_view == 'XZ':
                    self.ax.set_yticks([])
                    self.ax.yaxis.set_ticklabels([])
                    self.ax.view_init(elev=0, azim=90)

                elif self.selected_view == 'YZ':
                    self.ax.set_xticks([])
                    self.ax.xaxis.set_ticklabels([])
                    self.ax.view_init(elev=0, azim=0)

                self.update_plot_view(self.ax, self.selected_view)

    def update_plot_view(self, ax, selected_view):

        if selected_view == 'XY':
            ax.set_zticks([])
            ax.zaxis.set_ticklabels([])
            ax.view_init(elev=90, azim=0)

        elif selected_view == 'XZ':
            ax.set_yticks([])
            ax.yaxis.set_ticklabels([])
            ax.view_init(elev=0, azim=90)

        elif selected_view == 'YZ':
            ax.set_xticks([])
            ax.xaxis.set_ticklabels([])
            ax.view_init(elev=0, azim=0)

        ax.figure.canvas.draw_idle()
                
    def update_histogram_tab(self, fig):
        plot_area = self.plot_areas.get('Histogram')
        if plot_area:
            plot_area.clear_widgets()
            plot_area.add_widget(FigureCanvasKivyAgg(fig))

        else:
            self.add_message("Plot area not found.")

    def update_block_tab(self, fig, df=None):

        self.clear_plot_area()
        plot_area = self.plot_areas.get('Block')
        if plot_area:
            plot_area.clear_widgets()
            plot_widget = FigureCanvasKivyAgg(fig)
            plot_area.add_widget(plot_widget)
        else:
            self.add_message("Plot area not found")
    
    def update_plane_tab(self):
            print("update plane tab")

            self.processor.plane_tab(self.data_manager, self.selected_axis)

    def update_parameter_row(self, parameter_name, values):
        if parameter_name in self.parameter_rows:
            parameter_row = self.parameter_rows[parameter_name]

            values = values[::-1]
            while len(values) < 2:
                values.append(None)

            text_inputs = [child for child in parameter_row.children if isinstance(child, TextInput)]
            
            for i, text_input in enumerate(text_inputs):
                if values[i] is not None:
                    if isinstance(values[i], str):

                        text_input.text = values[i]
                    elif isinstance(values[i], (int, float)):

                        text_input.text = f"{values[i]:.2f}"
                    else:

                        text_input.text = ''
                else:

                    text_input.text = ''

        # print(f"Updated values for {parameter_name}: {values[::-1]}") 

    def update_marker_options(self, column_names):

        if hasattr(self, 'marker_spinner'):
            selected_columns = [column_names[5], column_names[7], column_names[13], column_names[18], column_names[24] ] if len(column_names) >= 23 else [] 

            formatted_headers = []

            for header in selected_columns:
                cleaned_header = header.replace('_x000D_', ' ').strip()
                formatted_header = ' '.join(cleaned_header.split('\n'))
                formatted_headers.append(formatted_header)

            self.marker_spinner.values = formatted_headers if formatted_headers else []

    def update_datetime_range_selector(self, min_datetime, max_datetime):
        if hasattr(self, 'datetime_range_selector'):

            self.datetime_range_selector.update_range(min_datetime, max_datetime)
            print("Datetime range selector updated with new range.")
        else:
            print("Datetime range selector not initialized.")

    def update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def add_message(self, message):
        # Schedule the message update to be run in the main thread
        Clock.schedule_once(lambda dt: self._update_message_text(message))

    def _update_message_text(self, message):
        if self.message_box.text:
            self.message_box.text += f"\n{message}"
        else:
            self.message_box.text = message

if __name__ == '__main__':
    GridAnalyzer().run()