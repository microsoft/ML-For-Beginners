import sys
from _pydevd_bundle import pydevd_xml
from os.path import basename
from _pydev_bundle import pydev_log
from urllib.parse import unquote_plus
from _pydevd_bundle.pydevd_constants import IS_PY311_OR_GREATER


#===================================================================================================
# print_var_node
#===================================================================================================
def print_var_node(xml_node, stream):
    name = xml_node.getAttribute('name')
    value = xml_node.getAttribute('value')
    val_type = xml_node.getAttribute('type')

    found_as = xml_node.getAttribute('found_as')
    stream.write('Name: ')
    stream.write(unquote_plus(name))
    stream.write(', Value: ')
    stream.write(unquote_plus(value))
    stream.write(', Type: ')
    stream.write(unquote_plus(val_type))
    if found_as:
        stream.write(', Found as: %s' % (unquote_plus(found_as),))
    stream.write('\n')


#===================================================================================================
# print_referrers
#===================================================================================================
def print_referrers(obj, stream=None):
    if stream is None:
        stream = sys.stdout
    result = get_referrer_info(obj)
    from xml.dom.minidom import parseString
    dom = parseString(result)

    xml = dom.getElementsByTagName('xml')[0]
    for node in xml.childNodes:
        if node.nodeType == node.TEXT_NODE:
            continue

        if node.localName == 'for':
            stream.write('Searching references for: ')
            for child in node.childNodes:
                if child.nodeType == node.TEXT_NODE:
                    continue
                print_var_node(child, stream)

        elif node.localName == 'var':
            stream.write('Referrer found: ')
            print_var_node(node, stream)

        else:
            sys.stderr.write('Unhandled node: %s\n' % (node,))

    return result


#===================================================================================================
# get_referrer_info
#===================================================================================================
def get_referrer_info(searched_obj):
    DEBUG = 0
    if DEBUG:
        sys.stderr.write('Getting referrers info.\n')
    try:
        try:
            if searched_obj is None:
                ret = ['<xml>\n']

                ret.append('<for>\n')
                ret.append(pydevd_xml.var_to_xml(
                    searched_obj,
                    'Skipping getting referrers for None',
                    additional_in_xml=' id="%s"' % (id(searched_obj),)))
                ret.append('</for>\n')
                ret.append('</xml>')
                ret = ''.join(ret)
                return ret

            obj_id = id(searched_obj)

            try:
                if DEBUG:
                    sys.stderr.write('Getting referrers...\n')
                import gc
                referrers = gc.get_referrers(searched_obj)
            except:
                pydev_log.exception()
                ret = ['<xml>\n']

                ret.append('<for>\n')
                ret.append(pydevd_xml.var_to_xml(
                    searched_obj,
                    'Exception raised while trying to get_referrers.',
                    additional_in_xml=' id="%s"' % (id(searched_obj),)))
                ret.append('</for>\n')
                ret.append('</xml>')
                ret = ''.join(ret)
                return ret

            if DEBUG:
                sys.stderr.write('Found %s referrers.\n' % (len(referrers),))

            curr_frame = sys._getframe()
            frame_type = type(curr_frame)

            # Ignore this frame and any caller frame of this frame

            ignore_frames = {}  # Should be a set, but it's not available on all python versions.
            while curr_frame is not None:
                if basename(curr_frame.f_code.co_filename).startswith('pydev'):
                    ignore_frames[curr_frame] = 1
                curr_frame = curr_frame.f_back

            ret = ['<xml>\n']

            ret.append('<for>\n')
            if DEBUG:
                sys.stderr.write('Searching Referrers of obj with id="%s"\n' % (obj_id,))

            ret.append(pydevd_xml.var_to_xml(
                searched_obj,
                'Referrers of obj with id="%s"' % (obj_id,)))
            ret.append('</for>\n')

            curr_frame = sys._getframe()
            all_objects = None

            for r in referrers:
                try:
                    if r in ignore_frames:
                        continue  # Skip the references we may add ourselves
                except:
                    pass  # Ok: unhashable type checked...

                if r is referrers:
                    continue

                if r is curr_frame.f_locals:
                    continue

                r_type = type(r)
                r_id = str(id(r))

                representation = str(r_type)

                found_as = ''
                if r_type == frame_type:
                    if DEBUG:
                        sys.stderr.write('Found frame referrer: %r\n' % (r,))
                    for key, val in r.f_locals.items():
                        if val is searched_obj:
                            found_as = key
                            break

                elif r_type == dict:
                    if DEBUG:
                        sys.stderr.write('Found dict referrer: %r\n' % (r,))

                    # Try to check if it's a value in the dict (and under which key it was found)
                    for key, val in r.items():
                        if val is searched_obj:
                            found_as = key
                            if DEBUG:
                                sys.stderr.write('    Found as %r in dict\n' % (found_as,))
                            break

                    # Ok, there's one annoying thing: many times we find it in a dict from an instance,
                    # but with this we don't directly have the class, only the dict, so, to workaround that
                    # we iterate over all reachable objects ad check if one of those has the given dict.
                    if all_objects is None:
                        all_objects = gc.get_objects()

                    for x in all_objects:
                        try:
                            if getattr(x, '__dict__', None) is r:
                                r = x
                                r_type = type(x)
                                r_id = str(id(r))
                                representation = str(r_type)
                                break
                        except:
                            pass  # Just ignore any error here (i.e.: ReferenceError, etc.)

                elif r_type in (tuple, list):
                    if DEBUG:
                        sys.stderr.write('Found tuple referrer: %r\n' % (r,))

                    for i, x in enumerate(r):
                        if x is searched_obj:
                            found_as = '%s[%s]' % (r_type.__name__, i)
                            if DEBUG:
                                sys.stderr.write('    Found as %s in tuple: \n' % (found_as,))
                            break

                elif IS_PY311_OR_GREATER:
                    # Up to Python 3.10, gc.get_referrers for an instance actually returned the
                    # object.__dict__, but on Python 3.11 it returns the actual object, so,
                    # handling is a bit easier (we don't need the workaround from the dict
                    # case to find the actual instance, we just need to find the attribute name).
                    if DEBUG:
                        sys.stderr.write('Found dict referrer: %r\n' % (r,))

                    dct = getattr(r, '__dict__', None)
                    if dct:
                        # Try to check if it's a value in the dict (and under which key it was found)
                        for key, val in dct.items():
                            if val is searched_obj:
                                found_as = key
                                if DEBUG:
                                    sys.stderr.write('    Found as %r in object instance\n' % (found_as,))
                                break

                if found_as:
                    if not isinstance(found_as, str):
                        found_as = str(found_as)
                    found_as = ' found_as="%s"' % (pydevd_xml.make_valid_xml_value(found_as),)

                ret.append(pydevd_xml.var_to_xml(
                    r,
                    representation,
                    additional_in_xml=' id="%s"%s' % (r_id, found_as)))
        finally:
            if DEBUG:
                sys.stderr.write('Done searching for references.\n')

            # If we have any exceptions, don't keep dangling references from this frame to any of our objects.
            all_objects = None
            referrers = None
            searched_obj = None
            r = None
            x = None
            key = None
            val = None
            curr_frame = None
            ignore_frames = None
    except:
        pydev_log.exception()
        ret = ['<xml>\n']

        ret.append('<for>\n')
        ret.append(pydevd_xml.var_to_xml(
            searched_obj,
            'Error getting referrers for:',
            additional_in_xml=' id="%s"' % (id(searched_obj),)))
        ret.append('</for>\n')
        ret.append('</xml>')
        ret = ''.join(ret)
        return ret

    ret.append('</xml>')
    ret = ''.join(ret)
    return ret

