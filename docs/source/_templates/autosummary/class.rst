{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% set base_members = [
   "get_params", "set_params", "score", "get_metadata_routing",
   "clone", "reset", "get_config", "set_config",
   "get_class_tag", "get_class_tags", "get_tag", "get_tags",
   "set_tags", "clone_tags", "get_test_params", "create_test_instance",
   "create_test_instances_and_names", "get_fitted_params", "is_composite",
   "get_param_names", "get_param_defaults", "save",
   "load_from_path", "load_from_serial",
] %}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% set ns = namespace(public=[]) %}
   {% for item in methods %}
   {% if not item.startswith('_') and not item.endswith('_request') and item not in base_members %}
   {% set ns.public = ns.public + [item] %}
   {% endif %}
   {% endfor %}
   {% if ns.public %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in ns.public %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set ns2 = namespace(public=[]) %}
   {% for item in attributes %}
   {% if not item.startswith('_') and item not in base_members %}
   {% set ns2.public = ns2.public + [item] %}
   {% endif %}
   {% endfor %}
   {% if ns2.public %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in ns2.public %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
