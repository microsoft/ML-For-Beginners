"""Terminal input and output prompts."""

from pygments.token import Token
import sys

from IPython.core.displayhook import DisplayHook

from prompt_toolkit.formatted_text import fragment_list_width, PygmentsTokens
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.enums import EditingMode


class Prompts(object):
    def __init__(self, shell):
        self.shell = shell

    def vi_mode(self):
        if (getattr(self.shell.pt_app, 'editing_mode', None) == EditingMode.VI
                and self.shell.prompt_includes_vi_mode):
            mode = str(self.shell.pt_app.app.vi_state.input_mode)
            if mode.startswith('InputMode.'):
                mode = mode[10:13].lower()
            elif mode.startswith('vi-'):
                mode = mode[3:6]
            return '['+mode+'] '
        return ''


    def in_prompt_tokens(self):
        return [
            (Token.Prompt, self.vi_mode() ),
            (Token.Prompt, 'In ['),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, ']: '),
        ]

    def _width(self):
        return fragment_list_width(self.in_prompt_tokens())

    def continuation_prompt_tokens(self, width=None):
        if width is None:
            width = self._width()
        return [
            (Token.Prompt, (' ' * (width - 5)) + '...: '),
        ]

    def rewrite_prompt_tokens(self):
        width = self._width()
        return [
            (Token.Prompt, ('-' * (width - 2)) + '> '),
        ]

    def out_prompt_tokens(self):
        return [
            (Token.OutPrompt, 'Out['),
            (Token.OutPromptNum, str(self.shell.execution_count)),
            (Token.OutPrompt, ']: '),
        ]

class ClassicPrompts(Prompts):
    def in_prompt_tokens(self):
        return [
            (Token.Prompt, '>>> '),
        ]

    def continuation_prompt_tokens(self, width=None):
        return [
            (Token.Prompt, '... ')
        ]

    def rewrite_prompt_tokens(self):
        return []

    def out_prompt_tokens(self):
        return []

class RichPromptDisplayHook(DisplayHook):
    """Subclass of base display hook using coloured prompt"""
    def write_output_prompt(self):
        sys.stdout.write(self.shell.separate_out)
        # If we're not displaying a prompt, it effectively ends with a newline,
        # because the output will be left-aligned.
        self.prompt_end_newline = True

        if self.do_full_cache:
            tokens = self.shell.prompts.out_prompt_tokens()
            prompt_txt = ''.join(s for t, s in tokens)
            if prompt_txt and not prompt_txt.endswith('\n'):
                # Ask for a newline before multiline output
                self.prompt_end_newline = False

            if self.shell.pt_app:
                print_formatted_text(PygmentsTokens(tokens),
                    style=self.shell.pt_app.app.style, end='',
                )
            else:
                sys.stdout.write(prompt_txt)

    def write_format_data(self, format_dict, md_dict=None) -> None:
        if self.shell.mime_renderers:

            for mime, handler in self.shell.mime_renderers.items():
                if mime in format_dict:
                    handler(format_dict[mime], None)
                    return
                
        super().write_format_data(format_dict, md_dict)

