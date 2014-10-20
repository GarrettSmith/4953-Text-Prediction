# Garrett Smith
# 3018390
#
# Machine Learning
# ACS-4953
#
# This is the demonstration application for the system.

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.listview import ListView
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.listview import ListItemLabel

from math import *
import itertools

from nltk import FreqDist

from controller import PredictionController

# How many predictions at most should we find
MAX_PREDICTIONS = 20

# Text input that can take a function to handle keyboard presses
class ShortcutTextInput(TextInput):
  def __init__(self, *args, **kwargs):
    self._keyboard_handler = kwargs.pop('keyboard_handler', None)
    super(ShortcutTextInput, self).__init__(*args, **kwargs)

  # Assign handler
  def set_keyboard_handler(self, func):
    self._keyboard_handler = func

  # Forward to handler if available
  def _keyboard_on_key_down(self, window, keycode, text, modifiers):
    if self._keyboard_handler is not None:
      # Fall through if not handled
      if not self._keyboard_handler(window, keycode, text, modifiers):
        super(ShortcutTextInput, self)._keyboard_on_key_down(window, keycode, text, modifiers)
    else:
      super(ShortcutTextInput, self)._keyboard_on_key_down(window, keycode, text, modifiers)

# Top level widget of the app
class InputScreen(GridLayout):

  # Create interface
  def __init__(self, **kwargs):
    self.controller = kwargs.pop('controller', None)
    super(InputScreen, self).__init__(**kwargs)
    self.cols = 2

    # Text input
    self.text = ShortcutTextInput(multiline=True, focus=True, keyboard_handler=self._on_keyboard_down)
    self.add_widget(self.text)
    self.text.bind(text=self._on_text)
    self.text.hint_text="Begin typing to test text prediction."

    # predictions
    obj_printer = lambda obj: "{} - {}".format(str(obj), self.current_prediction.get(obj, 0))
    args_converter = lambda row_index, obj: {'text': obj_printer(obj),
                                                   'size_hint_y': None,
                                                   'height': 25}

    self.prediction_adapter = ListAdapter(data=[],
                                               args_converter=args_converter,
                                               selection_mode='single',
                                               allow_empty_selection=False,
                                               cls=ListItemLabel)

    self.prediction_list = ListView(adapter=self.prediction_adapter)
    self.add_widget(self.prediction_list)

    self.current_prediction = FreqDist()

    # Record history
    self.previous_record = None
    self.in_word = False

  # Handle non-standard shortcuts
  # Mainly prediction selection and confirmation
  def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
    # print keycode
    if keycode[1] == 'tab':
      # Change selected prediction
      self.change_selection('shift' in modifiers)
      return True
    elif keycode[1] == 'up':
      self.change_selection(True)
      return True
    elif keycode[1] == 'down':
      self.change_selection(False)
      return True
    elif keycode[1] == 'enter':
      # Use the current prediction
      self._accept_prediction()
      return True
    elif keycode[1] == 'spacebar':
      # Record word
      self._accept_word(self._current_word())
      return False
    elif keycode[1] == 'backspace':
      # Remove previous word from record
      self.undo_word()
      return False
    else:
      # Default behaviour
      return  False  

  # Select a different prediction to use
  def change_selection(self, direction):  
    # Get current index
    selection = self.prediction_adapter.selection[0]
    index = self._selection_index(selection)

    # Deselect the current selection
    self._deselect_current()

    # whether to move back or forward
    if direction:
      index -= 1
    else:
      index += 1

    index = index % len(self.prediction_adapter.cached_views)

    # Select next
    self._select_at_index(index)

  # Get the index of the currently selected prediction
  def _selection_index(self, selection): 
    for i, view in self.prediction_adapter.cached_views.iteritems():
        if view == selection:
          return i

  # Deselect the currently selected prediction
  def _deselect_current(self):
    selection = self.prediction_adapter.selection[0]
    index = self._selection_index(selection)
    self.prediction_adapter.get_view(index).deselect()


  # Select the prediction with the given index
  def _select_at_index(self, index):
    view = self.prediction_adapter.get_view(index)
    view.select()
    self.prediction_adapter.selection[0] = view

  # Accept the selected prediction and insert it into the text field
  def _accept_prediction(self):
    offset = len(self._current_word())
    selection = self.prediction_adapter.selection[0]
    index = self._selection_index(selection)
    word = self.prediction_adapter.data[index]
    addition = word[offset:]
    self._accept_word(word)
    self.text.insert_text(addition + ' ')

  # Set that we are no longer within a word
  def _accept_word(self, word):
    self.in_word = False

  # Set that we are now in a word again after backspacing
  def undo_word(self):
    if self.previous_record is not None:
      if self.previous_record == self._current_word():
        self.previous_record = None
        self.in_word = True

  # Called when the text is changed to update the current predictions
  def _on_text(self, widget, text):
    # Record if we are typing a word
    if len(text) > 0 and text[-1] != ' ':
      self.in_word = True

    # Predict based on input
    prediction = self.controller.predict(str(text), self.in_word, MAX_PREDICTIONS)

    # Reset index if there is a prediction
    current_word = self._current_word()
    if len(current_word) > 0 and current_word not in prediction:
      prediction[current_word] = 0
    # populate list
    self.current_prediction = prediction
    self.prediction_adapter.data = self.controller.prediction_list(prediction)

    # Select a randomly select a weighted next word
    # rand = self.controller.weighted_random_prediction(prediction)
    # index = self.prediction_adapter.data.index(rand)
    # self._deselect_current()
    # self._select_at_index(index)

  # Returns the word currently being typed
  def _current_word(self):
    return self.text.text.split(' ')[-1]

# The app
class PredictionApp(App):

  # Create the application window
  def build(self):
    controller = PredictionController()
    return InputScreen(controller=controller)

# Run the app by default
if __name__ == '__main__':
  PredictionApp().run()