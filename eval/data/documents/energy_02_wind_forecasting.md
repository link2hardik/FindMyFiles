# Wind Power Forecasting

Wind generation forecasts estimate future electrical output from weather and turbine information. Forecasts may cover minutes, hours, or days, and each horizon supports different operational decisions.

A model can use wind speed, direction, air density, temperature, turbine availability, and recent power output. Forecast errors matter because grid operators must balance generation and demand continuously.

Evaluation should compare predictions with observed output using appropriate time-based splits rather than random splits that leak future conditions into training. Mean absolute error is easy to interpret, while probabilistic forecasts can report intervals that describe uncertainty. Weather and turbine changes should be recorded because they can explain unusual errors. Baselines such as persistence, which predicts that the next value resembles the latest value, help show whether a complex model adds value.

Errors should be examined by forecast horizon, wind regime, season, and turbine. A model that performs well on calm days may still be poor during rapidly changing weather, when accurate forecasts are most valuable to scheduling. The evaluation dataset should retain timestamps and the information that was available at prediction time so the experiment does not accidentally use future weather observations.