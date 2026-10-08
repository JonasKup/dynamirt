{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
{% if fullname == "dynamirt.fit.FitResult" %}
   :members: to_idata
{% elif fullname == "dynamirt.ModelContext" %}
   :members: design, index
{% endif %}
