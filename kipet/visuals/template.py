template_string = """
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
        <meta name="viewport" content="width=device-width,minimum-scale=1,initial-scale=1">
        <title>KIPET | {{ file_stem }} </title>
        <meta name="description" content="KIPET Report">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
        <!-- 
        <link rel="stylesheet" href="{{ base_style }}">
        <link rel="stylesheet" type="text/css" href="{{ code_style }}">
         -->
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css">
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        
        <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
           jax: ["input/TeX","output/HTML-CSS"],
            displayAlign: "left"
        });
        </script>
        <script type="text/javascript" id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
        </script>
        
        <style>
        {{ base_text }}
        {{ code_text }}
        {{ prism_text }}
        </style>
        
    </head>
    
    <body class="line-numbers">
       
        <div class="container"> <!-- -->
         {% set box_size = "28" %}
            <a id="top-page">
            <div class="container-left">
            <div class="d-flex flex-column flex-shrink-0 nav-left nav" style="width: 4.5rem;">
    
                <ul class="nav text-center">
                  <li class="nav-item">
                    <a href="#top-page" class="nav-link side-bar py-3" aria-current="page" title="Top" data-bs-toggle="tooltip" data-bs-placement="right">
                      <svg xmlns="http://www.w3.org/2000/svg" width={{ box_size }} height={{ box_size }} fill="currentColor" class="bi bi-arrow-up-square" viewBox="0 0 16 16">
                        <path fill-rule="evenodd" d="M15 2a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V2zM0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2zm8.5 9.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1 1-.708-.708l3-3a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 5.707V11.5z"/>
                      </svg>
                    </a>
                  </li>
                  <li>
                    <a id="model-go-to" href="#model-data-{{ default }}" class="nav-link side-bar py-3" title="Model Input" data-bs-toggle="tooltip" data-bs-placement="right">
                      <svg xmlns="http://www.w3.org/2000/svg" width={{ box_size }} height={{ box_size }} fill="currentColor" class="bi bi-sliders" viewBox="0 0 16 16">
                      <path fill-rule="evenodd" d="M11.5 2a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM9.05 3a2.5 2.5 0 0 1 4.9 0H16v1h-2.05a2.5 2.5 0 0 1-4.9 0H0V3h9.05zM4.5 7a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM2.05 8a2.5 2.5 0 0 1 4.9 0H16v1H6.95a2.5 2.5 0 0 1-4.9 0H0V8h2.05zm9.45 4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm-2.45 1a2.5 2.5 0 0 1 4.9 0H16v1h-2.05a2.5 2.5 0 0 1-4.9 0H0v-1h9.05z"/>
                    </svg>
                    </a>
                  </li>
                  <li>
                    <a id="results-go-to" href="#results-{{ default }}" class="nav-link side-bar py-3" title="Results" data-bs-toggle="tooltip" data-bs-placement="right">
                      <svg xmlns="http://www.w3.org/2000/svg" width={{ box_size }} height={{ box_size }} fill="currentColor" class="bi bi-graph-up" viewBox="0 0 16 16">
                      <path fill-rule="evenodd" d="M0 0h1v15h15v1H0V0zm10 3.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0V4.9l-3.613 4.417a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61L13.445 4H10.5a.5.5 0 0 1-.5-.5z"/>
                    </svg>
                    </a>
                  </li>
                
                {% if models|length > 1 %}
                <li>
                  <div class="dropstart">
                  <a class="nav-link side-bar py-3 drop" id="reaction-dropdown" data-bs-toggle="dropdown" title="Switch Experiment">
                    <svg xmlns="http://www.w3.org/2000/svg" width={{ box_size }} height={{ box_size }} fill="currentColor" class="bi bi-grid" viewBox="0 0 16 16">
                        <path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3zM2.5 2a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zm6.5.5A1.5 1.5 0 0 1 10.5 1h3A1.5 1.5 0 0 1 15 2.5v3A1.5 1.5 0 0 1 13.5 7h-3A1.5 1.5 0 0 1 9 5.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zM1 10.5A1.5 1.5 0 0 1 2.5 9h3A1.5 1.5 0 0 1 7 10.5v3A1.5 1.5 0 0 1 5.5 15h-3A1.5 1.5 0 0 1 1 13.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zm6.5.5A1.5 1.5 0 0 1 10.5 9h3a1.5 1.5 0 0 1 1.5 1.5v3a1.5 1.5 0 0 1-1.5 1.5h-3A1.5 1.5 0 0 1 9 13.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3z"/>
                    </svg>
                  </a>
                  <ul id="reaction-nav" class="dropdown-menu text-small shadow" role="tablist">
                   
                        <li class="nav-item side-bar-item">Reactions</li>
                        <li><hr class="dropdown-divider"></li>
                        {% for name in models.keys() %}
                            {% if loop.index == 1 %}
                                {% set active = 'active' %}
                            {% else %}
                                {% set active = '' %}
                            {% endif %}
                            <li class="nav-item {{ active }}"><a onclick="hrefUpdate(this)" class="side-bar-item" id="nav-tab-{{ name }}" role="tab" data-bs-toggle="tab" data-bs-target="#nav-{{ name }}" style="color:black;">{{ name }}</a></li>
                        {% endfor %}
                    
                  </ul>
                  </div>
                </li>
                {% endif %}
                  <li>
                    <a href="file:///{{ user_file }}" class="nav-link py-3 side-bar" title="Source File" data-bs-toggle="tooltip" data-bs-placement="right">
                      <svg xmlns="http://www.w3.org/2000/svg" width={{ box_size }} height={{ box_size }} fill="currentColor" class="bi bi-file-code" viewBox="0 0 16 16">
                        <path d="M6.646 5.646a.5.5 0 1 1 .708.708L5.707 8l1.647 1.646a.5.5 0 0 1-.708.708l-2-2a.5.5 0 0 1 0-.708l2-2zm2.708 0a.5.5 0 1 0-.708.708L10.293 8 8.646 9.646a.5.5 0 0 0 .708.708l2-2a.5.5 0 0 0 0-.708l-2-2z"/>
                        <path d="M2 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2zm10-1H4a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1z"/>
                      </svg>
                      </a>
                  </li>
                  <li class="nav-item">
                    <a href="https://github.com/salvadorgarciamunoz/kipet" class="nav-link py-3 side-bar" aria-current="page" title="Source Code on Github" data-bs-toggle="tooltip" data-bs-placement="right">
                      <svg xmlns="http://www.w3.org/2000/svg" width={{ box_size }} height={{ box_size }} fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
                      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    </a>
                  </li>
                  <li>
                    <a href="https://kipet.readthedocs.io" class="nav-link py-3 side-bar" title="Documentation" data-bs-toggle="tooltip" data-bs-placement="right">
                      <svg xmlns="http://www.w3.org/2000/svg" width={{ box_size }} height={{ box_size }} fill="currentColor" class="bi bi-question-circle" viewBox="0 0 16 16">
                      <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
                      <path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z"/>
                    </svg>
                    </a>
                  </li>
                  <li>
                    <a href="https://github.com/salvadorgarciamunoz/kipet/issues" class="nav-link py-3 side-bar" title="Report Bug" data-bs-toggle="tooltip" data-bs-placement="right">
                      <svg xmlns="http://www.w3.org/2000/svg" width={{ box_size }} height={{ box_size }} fill="currentColor" class="bi bi-bug" viewBox="0 0 16 16">
                          <path d="M4.355.522a.5.5 0 0 1 .623.333l.291.956A4.979 4.979 0 0 1 8 1c1.007 0 1.946.298 2.731.811l.29-.956a.5.5 0 1 1 .957.29l-.41 1.352A4.985 4.985 0 0 1 13 6h.5a.5.5 0 0 0 .5-.5V5a.5.5 0 0 1 1 0v.5A1.5 1.5 0 0 1 13.5 7H13v1h1.5a.5.5 0 0 1 0 1H13v1h.5a1.5 1.5 0 0 1 1.5 1.5v.5a.5.5 0 1 1-1 0v-.5a.5.5 0 0 0-.5-.5H13a5 5 0 0 1-10 0h-.5a.5.5 0 0 0-.5.5v.5a.5.5 0 1 1-1 0v-.5A1.5 1.5 0 0 1 2.5 10H3V9H1.5a.5.5 0 0 1 0-1H3V7h-.5A1.5 1.5 0 0 1 1 5.5V5a.5.5 0 0 1 1 0v.5a.5.5 0 0 0 .5.5H3c0-1.364.547-2.601 1.432-3.503l-.41-1.352a.5.5 0 0 1 .333-.623zM4 7v4a4 4 0 0 0 3.5 3.97V7H4zm4.5 0v7.97A4 4 0 0 0 12 11V7H8.5zM12 6a3.989 3.989 0 0 0-1.334-2.982A3.983 3.983 0 0 0 8 2a3.983 3.983 0 0 0-2.667 1.018A3.989 3.989 0 0 0 4 6h8z"/>
                      </svg>
                    </a>
                  </li>
                  
                </ul>
              </div>
            </div>
        
            
            <main role="main">
                <div class="d-flex flex-column flex-md-row align-items-center p-3 px-md-4 mb-3 bg-white border-bottom">
                    <h1 class="my-0 mr-md-auto font-weight-normal">KIPET Results
                      <!--
                      <img src="{{ logo_file }}" width="120" height="120" class="d-inline-block align-top" alt="">
                      -->
                    </h1>
                
                    <nav class="navbar navbar-expand-lg">
                        <div class="form-inline my-2 my-md-0">
                            <ul class="navbar-nav mr-auto">
                                <li class="nav-item">
                                    <a class="nav-link external" href="https://github.com/salvadorgarciamunoz/kipet">Github</a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link external" href="https://kipet.readthedocs.io">Documentation</a>
                                </li>
                            </ul>
                        </div>
                    </nav>
                </div>
            </main>
            
            <div class="section-block">
                <h2>Report</h2>
                <p>
                    This report has been automatically generated using KIPET (version {{ version }}).
                </p>
                {% if models|length > 1 %}
                <p>
                    These are the results from the solution of {{ models|length }} experiments or models using the ReactionSet class. Use the experiment button (<svg xmlns="http://www.w3.org/2000/svg" width="16px" height="16px" fill="currentColor" class="bi bi-grid" viewBox="0 0 16 16">
                        <path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h3A1.5 1.5 0 0 1 7 2.5v3A1.5 1.5 0 0 1 5.5 7h-3A1.5 1.5 0 0 1 1 5.5v-3zM2.5 2a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zm6.5.5A1.5 1.5 0 0 1 10.5 1h3A1.5 1.5 0 0 1 15 2.5v3A1.5 1.5 0 0 1 13.5 7h-3A1.5 1.5 0 0 1 9 5.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zM1 10.5A1.5 1.5 0 0 1 2.5 9h3A1.5 1.5 0 0 1 7 10.5v3A1.5 1.5 0 0 1 5.5 15h-3A1.5 1.5 0 0 1 1 13.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3zm6.5.5A1.5 1.5 0 0 1 10.5 9h3a1.5 1.5 0 0 1 1.5 1.5v3a1.5 1.5 0 0 1-1.5 1.5h-3A1.5 1.5 0 0 1 9 13.5v-3zm1.5-.5a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3z"/>
                    </svg>) to the left to switch between the results for each case.
                </p>
                {% else %}
                <p>
                    The results from the solution of a single experiment or model are presented below.
                </p>
                {% endif %}
            
                
            </div>
                
            <div class="tab-content" id="nav-tabContent">
                {% for name in models.keys() %}
                    {% if loop.index == 1 %}
                        {% set active = 'active' %}
                    {% else %}
                        {% set active = '' %}
                    {% endif %}
                <div class="tab-pane fade show {{ active }}" id="nav-{{ name }}" role="tabpanel">
                    
                    <div class="section-block">
                        <div class="info-block">
                            
                            <h3>General Information</h3>
                            <table class="table">
                                <caption>ReactionModel information</caption>
                                <tbody>
                                    <tr>
                                        <th width="15%" scope="row">Reaction(s)</th>
                                        <td>
                                            {% for name in models.keys() %}
                                                {{ name }} 
                                            {% endfor %}
                                        </td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Type</th>
                                        <td>{{ models[name].final_estimator }}</td>
                                    </tr>
                                    {% if models[name].opt_data.variance_used %}
                                    <tr>
                                        <th scope="row">Variance Method</th>
                                        {% if models[name].opt_data.var_method == 'originalchenetal' %}
                                            <td><a class="external" href="https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.2808">Chen et al. 2016 {{ models[name].opt_data.var_add }}</a></td>
                                        {% else %}
                                            <td><a class="external" href="https://www.sciencedirect.com/science/article/abs/pii/S0169743919307841">Short et al. 2020 {{ models[name].opt_data.var_add }}</a></td>
                                        {% endif %}
                                    </tr>
                                    {% endif %}
                                    <tr>
                                        <th scope="row">Date</th>
                                        <td>{{ models[name].time }}</td>
                                    </tr>
                                     <tr>
                                        <th scope="row">File</th>
                                        <td><a class="external" href="file:///{{ user_file }}">{{ user_file }}</a></td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Log</th>
                                        <td><a class="external" href="file:///{{ log_file }}">{{ log_file }}</a></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        
                        <a id="code-page">
                        <div class="info-block">
                            <h3>Code and Output</h3>
                            <p>To view the code used to generate these results as well as the output log, click the buttons below. This is to keep a record in case the files listed above have been modified or are no longer available.</p>
                            <p>
                                <a class="toggle" data-bs-toggle="modal" href="#collapse_code" role="button" aria-expanded="false" aria-controls="collapseExample">
                                   Show Code
                                </a>
                            </p>
                            <!-- <span style="white-space: pre-wrap;">{{ code_html }}</span> -->
                            
                            <a id="log-page">
                            
                            <p>
                                <a class="toggle" data-bs-toggle="modal" href="#collapse_log" role="button" aria-expanded="false" aria-controls="collapseExample">
                                   Show Log
                                </a>
                            </p>
                        
                        </div>
                        
                       
                        
                        
                        <div class="info-block">
                            
                        </div>
                        
                    </div>
                    
                    
                     
                    
                    
                    <a id="model-data-{{ name }}">
                    <div id="model-input" class="section-block">
                        <h2><i class="bi bi-sliders"></i>  Model <span style="color:gray; font-size: 18px;">({{ name }})</span></h2>
                        <div class="info-block">
                            <h3>Settings</h3>
                            <p>KIPET has many settings. If you wish to see the settings used in this current model, click on the button below.</p>
                            <p>
                                <a class="toggle" data-bs-toggle="collapse" href="#collapse_settings" role="button" aria-expanded="false" aria-controls="collapseExample">
                                   Show Settings
                                </a>
                            </p>
                    
                            <div id="collapse_settings" class="collapse">
                              
                                {% for key, value in models[name].settings.items() %}
                                <h4>{{ value[0] }} ({{ key }})</h4>
                                <table class="table">
                                    <tbody>
                                        <tr>
                                            <th width="15%" scope="col">Name</th>
                                            <th scope="col">Value</th>
                                        </tr>
                                        {% for key_, value_ in value[1].items() %}
                                        <tr>
                                            <td>{{ key_ }}</td>
                                            <td>{{ value_ }}</td>
                                        </tr>
                                        {% endfor %}
                                    
                                    </tbody>
                                </table>
                                {% endfor %}
                                <p>
                                    <a class="toggle" data-bs-toggle="collapse" href="#collapse_settings" role="button" aria-expanded="false" aria-controls="collapseExample" onclick="MoveUp()">
                                       Close Settings
                                    </a>
                                </p>
                            </div>
                        </div>
                        
                        <div class="info-block">  
                            <h3>Model Parameters</h3>
                            {% if models[name].param_data|length == 0 %}
                            <p>The current model does not have any parameter information</p>
                            {% else %}
                            <table class="table">
                                <caption>Model parameters their provided data.</caption>
                                <thead>
                                    <tr>
                                        <th width="10%" scope="col">Name</th>
                                        <th scope="col">Value</th>
                                        <th scope="col">Bounds</th>
                                        <th scope="col">Units</th>
                                        <th scopr="col">Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for param in models[name].param_data %}
                                    <tr>
                                        <td>{{ param.name }}</td>
                                        <td>{{ param.initial }}</td>
                                        <td>({{ param.lb}}, {{ param.ub }})</td>
                                        <td>{{ param.units }}</td>
                                        <td>{{ param.description }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% endif %}
                        </div>
                        
                        <div class="info-block">
                            <h3>Model Components</h3>
                            {% if models[name].comp_data|length == 0 %}
                            <p>The current model does not have any components</p>
                            {% else %}
                            <table class="table">
                                <caption>Model components and their provided data.</caption>
                                <thead>
                                    <tr>
                                        <th width="10%" scope="col">Name</th>
                                        <th scope="col">Initial Value</th>
                                        <th scope="col">Units</th>
                                        <th scope="col">Variance</th>
                                        <th scope="col">Known I.C.</th>
                                        <th scope="col">Absorbing</th>
                                        <th scope="col">Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for comp in models[name].comp_data %}
                                    <tr>
                                        <td>{{ comp.name }}</td>
                                        <td>{{ comp.value }}</td>
                                        <td>{{ comp.units }}</td>
                                        <td>{{ comp.variance }}</td>
                                        <td>{{ comp.known }}</td>
                                        <td>{{ comp.absorbing }}</td>
                                        <td>{{ comp.description }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% endif %}
                        </div>
                        
                        <div class="info-block">
                            <h3>Model States</h3>
                            {% if models[name].state_data|length == 0 %}
                            <p>The current model does not have any complementary states.</p>
                            {% else %}
                            <table class="table">
                                <caption>Model states and their provided data. Note that volume (V) is added automatically to all models.</caption>
                                <thead>
                                    <tr>
                                        <th width="10%" scope="col">Name</th>
                                        <th scope="col">Initial Value</th>
                                        <th scope="col">Units</th>
                                        <th scope="col">Variance</th>
                                        <th scope="col">Known I.C.</th>
                                        <th scope="col">Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for comp in models[name].state_data %}
                                    <tr>
                                        <td>{{ comp.name }}</td>
                                        <td>{{ comp.value }}</td>
                                        <td>{{ comp.units }}</td>
                                        <td>{{ comp.variance }}</td>
                                        <td>{{ comp.known }}</td>
                                        <td>{{ comp.description }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% endif %}
                        </div>
                        
                        <div class="info-block">
                            <h3>Model Constants</h3>
                            {% if models[name].const_data|length == 0 %}
                            <p>The current model does not have any constants</p>
                            {% else %}
                            <table class="table">
                                <caption>Model constants and their provided data.</caption>
                                <thead>
                                    <tr>
                                        <th width="10%" scope="col">Name</th>
                                        <th scope="col">Value</th>
                                        <th scope="col">Units</th>
                                        <th scope="col">Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for comp in models[name].const_data %}
                                    <tr>
                                        <td>{{ comp.name }}</td>
                                        <td>{{ comp.value }}</td>
                                        <td>{{ comp.units }}</td>
                                        <td>{{ comp.description }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% endif %}
                        </div>
                        
                        <div class="info-block">
                            <h3>ODEs</h3>
                            <div class="big-equations math-left-align">
                                <table class="table">
                                    <caption>ODEs used in the ReactionModel</caption>
                                    <thead>
                                        <tr>
                                            <th width="10%" scope="col">Variable</th>
                                            <th scope="col">Equation</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for key, value in models[name].ode_data.items() %}
                                        <tr>
                                            <td>{{ key }}</td>
                                            <td>{{ value }}</td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <div class="info-block">
                            <h3>DAEs</h3>
                            {% if models[name].age_data is not none %}
                            <div class="big-equations">
                                <table class="table">
                                    <caption>DAEs used in the ReactionModel</caption>
                                    <thead>
                                        <tr>
                                            <th width="10%" scope="col">Variable</th>
                                            <th width="10%" scope="col">Reaction</th>
                                            <th scope="col">Equation</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for key, value in models[name].age_data.items() %}
                                        <tr>
                                            <td>{{ key }}</td>
                                            <td>{{ value[0] }}</td>
                                            <td>{{ value[1] }}</td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                            {% else %}
                            <p>The current model does not have any DAEs</p>                
                            {% endif %}  
                        </div>
                        
                        <div class="info-block">
                            <h3>Dosing Points</h3> 
                            {% if models[name].feeds is none %}
                            <p>The current model does not have any dosing points</p>
                            {% else %}
                            <table class="table">
                                <caption>Dosing point information</caption>
                                <thead>
                                    <tr>
                                        <th width="10%" scope="col">Component</th>
                                        <th scope="col">Time</th>
                                        <th scope="col">Concentration</th>
                                        <th scope="col">Amount</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for comp in models[name].feeds %}
                                    <tr>
                                        <td>{{ comp[0] }}</td>
                                        <td>{{ comp[1] }}</td>
                                        <td>{{ comp[2][0] }} {{ comp[2][1] }}</td>
                                        <td>{{ comp[3][0] }} {{ comp[3][1] }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% endif %}
                        </div>
                        
                        {% if models[name].abs_data is not none %}
                        <div class="info-block">
                            <h3>Absorbance Data</h3>
                            <p> Provided absorbance data for the following components has been provided:</p>
                            {% for chart in models[name].chart_abs_files %}
                            <div class="chart">
                                <iframe width="100%" height="550px" frameborder="0" seamless="seamless" scrolling="no" src="{{ chart }}"></iframe>
                            </div>
                            <a class="external" href="{{ chart }}">File</a>
                            {% endfor %}
                        {% endif %}
                    
                        {% if models[name].data_chart_files is not none %}
                        <div  class="info-block">
                            <h3>Spectral Data</h3>
                            <table class="table">
                                <tbody>
                                    <tr>
                                        <th scope="row">File</th>
                                        <td><a class="external" href="file:///{{ models[name].spectra_file }}">{{ models[name].spectra_file }}</a></td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Unwanted Contribution</th>
                                        <td>{{ models[name].g_contrib }}</td>
                                </tbody>
                            </table>
                            <h4>Pre-processing</h4>
                            {% if models[name].spectral_info is not none %}
                            {% for sd in models[name].spectral_info %}
                            <table class="table">
                                <tbody>
                                {% for key, value in sd.items() %}
                                    <tr>
                                        <th width="15%" scope="row">{{ key }}</th>
                                        <td>{{ value }}</td>
                                    </tr>
                                {% endfor %}
                                </tbody>
                            </table>
                            {% endfor %}
                            {% endif %}
                            
                            <a id="spectral-page">
                            <p>
                                <a class="toggle" data-bs-toggle="collapse" href="#collapse_D" role="button">
                                    Show Spectral Data <i class="bi bi-graph-up"></i>
                                </a>
                            </p>

                            <div class="collapse" id="collapse_D">
                                <div class="chart">
                                    <iframe width="100%" height="550px" frameborder="0" seamless="seamless" src="{{ models[name].data_chart_files }}"></iframe>
                                </div>
                                <a class="external file" href="{{ models[name].data_chart_files }}">File</a>
                            </div>
                        </div> 
                        {% endif %}
                    </div> <!-- section block -->
                
                    <a id="results-{{ name }}">
                    <div id="results" class="section-block">
                        <h2 class="mt-2 mb-2"><i class="bi bi-graph-up"></i>  Results  <span style="color:gray; font-size: 18px;">({{ name }})</span></h2>
                      
                        {% if models[name].final_estimator == 'parameter estimation' %}
                        <div class="info-block">  
                            <h3>Parameter Results</h3>
                            {% if models[name].param_data|length == 0 %}
                            <p>The current model does not have any parameter information</p>
                            {% else %}
                            <table class="table">
                                <caption>Parameters fit using KIPET alongside their provided data.</caption>
                                <thead>
                                    <tr>
                                        <th width="10%" scope="col">Name</th>
                                        <th scope="col">Optimal Value</th>
                                        
                                        {% if models[name].confidence is not none %}
                                        <th scope="col">+/- (95% CI)</th>
                                        {% endif %}
                                        
                                        <th scope="col">Bounds</th>
                                        <th scope="col">Units</th>
                                        <th scopr="col">Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for param in models[name].param_data %}
                                    
                                    <tr>
                                        <td>{{ param.name }}</td>
                                        <td>{{ "%.8e"|format(param.value) }}</td>
                                        
                                        {% if models[name].confidence is not none %}
                                        <td>
                                            {% if not param.fixed %}
                                            {{ "%.8e"|format(models[name].confidence[param.name]) }}
                                            {% else %}
                                            -
                                            {% endif %}
                                        </td>
                                        {% endif %}
                                        
                                        <td>({{ param.lb}}, {{ param.ub }})</td>
                                        <td>{{ param.units }}</td>
                                        <td>{{ param.description }}</td>
                                    </tr>
                                    
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% endif %}
                        </div>
                        {% endif %}
                        
                        
                        {% if models[name].covariance is not none %}
                        <div class="info-block">  
                            <h3>Parameter Covariances</h3>
                            
                            <table class="table">
                                <caption>Parameter covariances predicted from inverse reduced Hessian.</caption>
                                <thead>
                                    <tr>
                                    <th scope="col"></th>
                                    {% for param in models[name].covariance.columns %}
                                        <th scope="col">{{ param }}</th>
                                    {% endfor %}
                                    </tr>
                                </thead>
                        
                                <tbody>
                                    {% for param in models[name].covariance.columns %}
                                    <tr>
                                        <th scope="row">{{ param }}</th>
                                        {% for param2 in models[name].covariance.columns %}
                                            <td> {{ "%.8e"|format(models[name].covariance[param][param2]) }}</td>
                                        {% endfor %}
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            
                        </div>
                        {% endif %}
                        
                        
                        {% if models[name].final_estimator != 'simulation' %}
                        <div class="info-block">  
                            <h3>Component Variances</h3>
                            {% if models[name].variances|length == 0 %}
                            <p>The current model does not have any variance information</p>
                            {% else %}
                            <table class="table">
                                {% if models[name].opt_data.variance_used %}
                                    {% if models[name].opt_data.var_method == 'originalchenetal' %}
                                        <caption>Variances calculated using the method in <a class="external" href="https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.2808">Chen et al. 2016</i></a></caption>
                                    {% else %}
                                        <caption>Variances calculated using the method in <a class="external" href="https://www.sciencedirect.com/science/article/abs/pii/S0169743919307841">Short et al. 2020</a></caption>
                                    {% endif %}
                                {% else %}
                                    <caption>Variances provided by the user</caption>
                                {% endif %}
                                <thead>
                                    <tr>
                                        <th width="10%" scope="col">Name</th>
                                        <th scope="col">Variance</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for param, variance in models[name].variances.items() %}
                                    <tr>
                                        <td>{{ param }}</td>
                                        <td>{{ "%.5e"|format(variance) }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                            {% endif %}
                        </div>
                        {% endif %}
                        
                        {% if models[name].opt_data.var_method == 'direct_sigmas' %}
                        <div class="info-block">  
                            <h3>Calculated Variances</h3>
                            <p>These values were calculated using the direct sigma method. (<a class="external" href="https://www.sciencedirect.com/science/article/abs/pii/S0169743919307841">Short et al. 2020</a>)</p>
                        
                            {% for key, value in models[name].delta_results.items() %}
                             <h4>Point #{{ key }}</h4>
                             <table class="table">
                                 <tbody>
                                     <tr>
                                         <th width="15%" scope="col">Name</th>
                                         <th scope="col">Value</th>
                                     </tr>
                                     {% for key_, value_ in value.items() %}
                                     <tr>
                                         <td>{{ key_ }}</td>
                                         <td>{{ value_ }}</td>
                                     </tr>
                                     {% endfor %}
                                 
                                 </tbody>
                             </table>
                             {% endfor %}
                        </div>
                        {% endif %}
                        
                        {% if models[name].bounds is not none and models[name].bounds|length > 0 %}
                        <div class="info-block"> 
                            <h3>Profile Bounds</h3>
                            <table class="table">
                                <caption>User defined profile bounds</caption>
                                <thead>
                                    <tr>
                                        <th width="10%" scope="col">Name</th>
                                        <th scope="col">Profile</th>
                                        <th scope="col">Range</th>
                                        <th scopr="col">Bounds</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for stat in models[name].bounds %}
                                    <tr>
                                        <td>{{ stat[1] }}</td>
                                        <td>{{ stat[0] }}</td>
                                        <td>{{ stat[2] }}</td>
                                        <td>{{ stat[3] }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% endif %}
                        
                        <div class="info-block">
                            <h3>Charts</h3>
                            {% for chart in models[name].chart_C_files %}
                            <div class="chart">
                                <iframe width="100%" height="550px" frameborder="0" seamless="seamless" src="{{ chart }}"></iframe>
                            </div>
                            <a class="external file" href="{{ chart }}">File</a>
                            {% endfor %}
                            
                            {% for chart in models[name].chart_S_files %}
                            <div class="chart">
                                <iframe width="100%" height="550px" frameborder="0" seamless="seamless" src="{{ chart }}"></iframe>
                            </div>
                            <a class="external file" href="{{ chart }}">File</a>
                            {% endfor %}
                            
                            {% for chart in models[name].chart_U_files %}
                            {% if loop.index == 1 %}
                            <p class="mt-5">
                                <a class="toggle" data-bs-toggle="collapse" href="#collapse_U" role="button" aria-expanded="false" aria-controls="collapseExample">
                                    Show State Charts <i class="bi bi-graph-up"></i>
                                </a>
                            </p>
                            {% endif %}
                            <div class="collapse" id="collapse_U">
                                <div class="chart">
                                    <iframe width="100%" height="550px" frameborder="0" seamless="seamless" src="{{ chart }}"></iframe>
                                </div>
                                <a class="external file" href="{{ chart }}">File</a>
                            </div>
                            {% endfor %}
                            
                            {% for chart in models[name].chart_Y_files %}
                            {% if loop.index == 1 %}
                            <p>
                                <a class="toggle" data-bs-toggle="collapse" href="#collapse_Y" role="button" aria-expanded="false" aria-controls="collapseExample">
                                    Show DAE Charts <i class="bi bi-graph-up"></i>
                                </a>
                            </p>
                            {% endif %}
                            <div class="collapse" id="collapse_Y">
                                <div class="chart">
                                    <iframe width="100%" height="550px" frameborder="0" seamless="seamless" src="{{ chart }}"></iframe>
                                </div>
                                <a class="external file" href="{{ chart }}">File</a>
                            </div>
                            {% endfor %}
                        </div>
                        
                        {% if models[name].final_estimator == 'parameter estimation' %}
                        <div class="info-block"> 
                            <h3>Diagnostics</h3>
                            <p>
                                <a class="toggle" data-bs-toggle="collapse" href="#collapse_fit" role="button" aria-expanded="false" aria-controls="collapseExample">
                                   Show Diagnostics
                                </a>
                            </p>
                            <div id="collapse_fit" class="collapse">
                                <table class="table">
                                    <caption>Model statistics related to fit.</caption>
                                    <thead>
                                        <tr>
                                            <th scope="col">Statistic</th>
                                            <th scope="col">Value</th>
                                            <th scopr="col">Description</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for stat in models[name].diagnostics %}
                                        <tr>
                                            <td>{{ stat.name }}</td>
                                            <td>{{ "%.6e"|format(stat.value) }}</td>
                                            <td>{{ stat.description }}</td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                                {% if models[name].res_chart is not none %}
                                <div class="chart">
                                    <iframe id="i1" width="100%" height="550px" frameborder="0" seamless="seamless" src="{{ models[name].res_chart }}"></iframe>
                                </div>
                                <a class="external file" href="{{ models[name].res_chart }}">File</a>
                                {% endif %}
                                
                                {% if models[name].par_chart is not none %}
                                <div class="chart">
                                    <iframe width="100%" height="100%" frameborder="0" seamless="seamless" src="{{ models[name].par_chart }}"></iframe>
                                </div>
                                <a class="external file" href="{{ models[name].par_chart }}">File</a>
                                {% endif %}
                                <p class="mt-4">
                                    <a class="toggle" data-bs-toggle="collapse" href="#collapse_fit" role="button" aria-expanded="false" aria-controls="collapseExample">
                                       Close Diagnostics
                                    </a>
                                </p>
                            </div>
                        </div>
                        {% endif %}
                    </div>

                </div>
                {% endfor %}
            </div>
        </div> <!-- container -->

          <!-- Modal -->
          <div class="modal fade" id="collapse_code" tabindex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-dialog-scrollable" role="document">
              <div class="modal-content">
                <div class="modal-header">
                  <h5 class="modal-title" id="exampleModalLabel">Source Code: {{ file_stem }}</h5>
                  <button type="button" class="btn-danger" data-bs-dismiss="modal" aria-label="Close"><i class="bi bi-x-lg"></i></button>
                </div>
                <div class="modal-body">
                 <div style="padding: 16px;">
                                  <pre style="border-radius: 12px;">
<code class="language-python">
{{ source_code }}                
</code>
                                  </pre>
                              </div>
                </div>
                <div class="modal-footer">
                  <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Modal -->
          <div class="modal fade" id="collapse_log" tabindex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-dialog-scrollable" role="document">
              <div class="modal-content">
                <div class="modal-header">
                  <h5 class="modal-title" id="exampleModalLabel">Log: {{ file_stem }}</h5>
                  <button type="button" class="btn-danger" data-bs-dismiss="modal" aria-label="Close"><i class="bi bi-x-lg"></i></button>
                </div>
                <div class="modal-body">
                 <div style="padding: 16px;">
                                  <pre style="border-radius: 12px;">
<code class="language-*">
{{ log }}                
</code>
                                  </pre>
                              </div>
                </div>
                <div class="modal-footer">
                  <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
              </div>
            </div>
          </div>


    </body>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
       function MoveUp() {
          var top = $('#model-input').position().top;
          $(window).scrollTop( top );
       }
       function MoveUpCode() {
          var top = $('#code-page').position().top;
          $(window).scrollTop( top );
       }
       function MoveUpLog() {
          var top = $('#log-page').position().top;
          $(window).scrollTop( top );
       }
       function MoveUpFit() {
          var top = $('#log-page').position().top;
          $(window).scrollTop( top );
       }
       
    </script>   
    
    <script type="text/javascript">
        var hrefUpdate = function(e) {
        var aId = e.id;
        terms = aId.split("-")
        newName = `${terms[2]}-${terms[3]}`
        document.getElementById("results-go-to").href=`#results-${newName}`; 
        document.getElementById("model-go-to").href=`#model-data-${newName}`; 
        return false;
        };
    </script>
       
    <script>
       $('#reaction-nav a').on('click', function(e) {
        e.preventDefault();
        $(this).tab('show');
        var theThis = $(this);
        $('#reaction-nav a').removeClass('active');
        theThis.addClass('active');
        });
       
    </script>
    
    <script>
        {{ prism_js }}
    </script>
    
</html>

"""