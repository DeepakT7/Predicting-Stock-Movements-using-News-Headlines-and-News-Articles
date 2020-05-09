##from .blueprints.home.views import home_blueprint
##from flask_babel import Babel, _
##
##app.register_blueprint(home_blueprint, url_prefix='/<lang_code>')
##
##babel = Babel()
##
##@babel.localeselector
##def get_locale():
##    request_lc = request.args.get('lc')
##    if not request_lc:
##        if not 'lang_code' in g:
##            # use default
##            g.lang_code = 'en'
##            request_lc = 'en_US'
##        else:
##            if g.lang_code == 'es':
##                request_lc = 'es_ES'
##            elif g.lang_code == 'nl':
##                request_lc = 'nl_NL'
##            else:
##                request_lc = 'en_US'
##
##    else:
##        # set g.lang_code to the requested language
##        if request_lc == 'nl_NL':
##           g.lang_code = 'nl'
##        elif request_lc == 'es_ES':
##           g.lang_code = 'es'
##        else:
##            request_lc = 'en_US'
##            g.lang_code = 'en'
##        #sys.exit()
##    session['lc'] = request_lc
##    return request_lc
