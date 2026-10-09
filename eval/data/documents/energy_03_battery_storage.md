# Battery Energy Storage

Battery storage can shift electricity from one time to another. It may charge when renewable generation is abundant or prices are low and discharge during periods of high demand.

A battery system has limits including energy capacity, maximum charge and discharge power, round-trip efficiency, temperature range, and degradation. A control strategy that ignores these limits may look good in a simulation but fail in operation.

A storage evaluation should measure more than total energy moved. Useful measures include peak reduction, renewable curtailment, cost savings, state-of-charge violations, cycle count, and battery lifetime assumptions.

Charging from a clean source does not automatically make every storage schedule beneficial. Round-trip losses, battery degradation, and the carbon intensity of grid electricity should be included in comparisons. A controller that reduces a monthly bill may increase wear or shift emissions into a dirtier period.

Simulation inputs should be kept separate from results and should include their time resolution. A one-hour model can miss short peaks that require high power, while a very detailed model may be expensive to run. Sensitivity analysis can show whether a conclusion depends on uncertain battery price, efficiency, renewable output, or electricity tariffs.