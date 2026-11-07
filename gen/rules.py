"""Hand-written readout: concepts (L8) and one neuron per handwriting style (L9).

Every rule describes how people write the character, in terms of named features.
Nothing here was fitted. Syntax of a style rule:

    "feat*w feat ?optional_feat ... | contradiction*w ..."

- Positive weights are normalised to sum to 1: a style scores 1.0 when everything it expects is there.
- Rules are closed-world: every SALIENT feature that the rule neither expects, allows (?), nor
  lists as a contradiction subtracts UNEXP when present. "An E with a loop is not an E."
- Listed contradictions subtract NEG * w.

Feature names: <map>@<zone>. Zones: A (all), T/M/B (top/middle/bottom rows), l/c/r (columns),
or both (Tl = top-left).
  edge_l / edge_r      vertical stroke with nothing to its left / right (stems, sides)
  edge_t / edge_b      horizontal stroke with nothing above / below it
  mid_h / mid_v        horizontal stroke with ink above and below / vertical with ink both sides
  run_d1 / run_d2      long diagonal stroke / or \\
  end_any              a stroke ends here (end_up, end_dl, ... for a direction)
  bay_R/L/U/D          background walled on three sides, open to the right/left/up/down
  loop, loop_two, loop_tall, loop_hi, loop_lo, gap, narrow   (see CONCEPTS and generate.py)
"""

NEG = 0.5
UNEXP = 0.3

SALIENT = ["loop", "loop_two", "gap", "mid_h@M",
           "bay_R@T", "bay_R@M", "bay_R@B", "bay_L@T", "bay_L@M", "bay_L@B",
           "bay_U@T", "bay_U@B", "bay_D@T", "bay_D@B",
           "end_any@Tl", "end_any@Tr", "end_any@Bl", "end_any@Br"]
# allowing a loop also allows its kind
IMPLIES = {"loop": ["loop_two"], "loop_two": ["loop"], "loop_tall": ["loop"], "loop_hi": ["loop"], "loop_lo": ["loop"]}

# L8 concepts: name -> ("feat*w feat*w ...", bias, why)
CONCEPTS = {
    "loop_tall": ("hole@T hole@B waist*-2", -1, "one loop reaching both top and bottom (0, O, D)"),
    "loop_two": ("hole@T hole@B waist", -2, "two loops stacked with a stroke between them (8, B)"),
    "narrow": ("wide_l*-1 wide_r*-1", 1, "ink stays near the centre column (1, l, I, i)"),
    "stem_l": ("edge_l@T edge_l@B", -1, "a vertical stroke down the whole left side"),
    "stem_r": ("edge_r@T edge_r@B", -1, "a vertical stroke down the whole right side"),
    "a_body": ("loop_left vx1@r end_any@Br", -1.5, "an a: a bowl to the left of a straight stem that ends in a tail at the bottom right (all three together)"),
    "tail_below_r": ("loop_hi hx1@D marg_l@B", -1.5, "a bowl at the top with one stroke hanging below it on the right, leaving the bottom left empty (q, g)"),
    "tail_below_l": ("loop_hi hx1@D marg_r@B", -1.5, "a bowl at the top with one stroke hanging below it on the left, leaving the bottom right empty (p)"),
    "B_body": ("run_v@l hole@T hole@B waist", -3, "a B: two bowls stacked with a waist between them, on a straight stem down the left (all together)"),
    "b_body": ("asc_l@A loop_lo", -1, "a b: a bowl low down, its stem continuing above it on the bowl's left (wherever the letter sits)"),
    "d_body": ("asc_r@A loop_lo", -1, "a d: a bowl low down, its stem continuing above it on the bowl's right (wherever the letter sits)"),
    "D_body": ("stem_lh@T stem_lh@B hole@T hole@B waist*-1", -3, "a D: one tall bowl with a straight stem beside it from the top band to the bottom band"),
    "E_arms": ("bay_R@T bay_R@B end_any@M", -2, "an E: two openings to the right, one above and one below a middle arm that ends in the middle (wherever the letter sits)"),
    "e_body": ("loop_hi bay_R@A", -1, "an e: a loop at the top and an opening to the right below it"),
    "S_body": ("run_d2@M ax2@M", -1, "an S: its spine runs down to the right through the middle, and the / lines there cross it twice (wherever it sits)"),
    "line": ("run_v@A run_d1@A run_d2@A", 0, "a long straight stroke in any direction (one is enough)"),
}

# a single straight line, upright or slanted: one end at the top, one at the bottom, nothing else
STICK = ("line*2 end_any@T end_any@B ?narrow ?stem_l ?stem_r ?end_any@Tl ?end_any@Tr ?end_any@Bl ?end_any@Br "
         "| loop*2 gap*2 edge_t@T*2 edge_b@B*2 run_h@T run_h@B run_h@M*2 ink@Ml*2 ink@Mr*2 junction@A*1.5 "
         "bay_L@T*2 bay_L@M*2 bay_L@B*2 bay_R@T*2 bay_R@M*2 bay_R@B*2 bay_U@T*2 bay_D@B*2")
# one big interior; an overlap where the pen started can add a tiny second hole
ROUND = "hole_big*3 loop ?loop_tall ?loop_lo ?loop_hi ?narrow ?mid_h@M | waist*1.5"
OPEN_R = "bay_R@T bay_R@M*2 bay_R@Mc bay_R@B end_any@Tr end_any@Br edge_b@B ?edge_t@T | loop"
SNAKE = ("S_body bay_R@T*2 bay_L@B*2 end_any@Tr end_any@Bl ?bay_R@M ?bay_L@M ?mid_h@M ?end_any@Tc ?end_any@Bc "
         "| loop stem_l*0.5")
CROSS = ("bay_U@T bay_D@B bay_R@M bay_L@M end_any@Tl end_any@Tr end_any@Bl end_any@Br end_ul@T "
         "?bay_R@T ?bay_R@B ?bay_L@T ?bay_L@B ?crossing@M ?junction@M | loop*2 junction@Ml*0.5 stem_l")
VEE = "bay_U@T*2 bay_U@M*2 run_d1@A run_d2@A end_any@Tl end_any@Tr ?bay_U@B ?end_any@Bl ?end_any@Br | loop*2 stem_l stem_r hx3@M*2"
# W: two openings at the top, one at the bottom centre. M is its mirror image.
DOUBLE_V = ("bay_U@Tl bay_U@Tr bay_D@Bc hx3@M*3 run_d1@A run_d2@A end_any@Tl end_any@Tr ?end_any@Tc ?end_any@Bl "
            "?end_any@Br ?bay_U@T ?bay_D@B ?mid_v@M ?mid_h@M ?edge_t@M | loop*2 edge_t@T*1.5")
EM = ("bay_U@Tc*2 bay_D@Bl bay_D@Br run_d1@A run_d2@A end_any@Bl end_any@Br ?end_any@Tl ?end_any@Tr ?end_any@Tc "
      "?end_any@Bc ?bay_U@T ?bay_D@B ?mid_v@M ?mid_h@M | loop*2 bay_U@Tr*1.5")
ZED = ("edge_t@T*2 edge_b@B*2 run_d1@M*2 bay_L@T bay_L@M bay_R@M bay_R@B end_any@Tl end_any@Br "
       "?end_any@Bl ?end_any@Tr ?bay_L@B ?bay_R@T | loop*2 run_d2@M*0.5")

STYLES = {
    # ---- digits -------------------------------------------------------------
    "0/oval": ROUND,
    "1/stick": STICK,
    "1/flag": "line*2 end_any@Tl*2 end_any@B ?narrow ?stem_r ?stem_l ?end_any@Tc ?end_any@Br ?end_any@Bl ?bay_L@M ?bay_L@B | loop*2 gap*2 edge_b@B edge_t@T*2 run_h@T*2",
    "1/flag_base": "end_any@Tl*2 edge_b@B bay_L@M end_any@Bl end_any@Br ?bay_L@B ?end_any@Tc ?narrow | loop*2 edge_t@T*2 run_h@T*2",
    "2/curve": "bay_L@T bay_L@M bay_R@M bay_R@B*2 edge_b@B*2 end_any@Tl end_any@Br*2 ?end_any@Tr ?end_any@Bl ?bay_D@B ?mid_h@M | loop*2 bay_L@B*1.5",
    "2/loop": "bay_L@T bay_L@M bay_R@M bay_R@B*2 edge_b@B*2 end_any@Tl end_any@Br*2 ?loop ?end_any@Tr ?end_any@Bl ?bay_L@B ?bay_D@B ?mid_h@M",
    "3/round": "bay_L@T*2 bay_L@M bay_L@B*2 edge_b@B*2 ink@Bl edge_r@Mr end_any@Tl end_any@Bl ?end_any@Tr ?end_any@Ml ?end_any@Mc ?mid_h@M ?ink@Br | loop*2 bay_R@T bay_R@B bay_R@M end_down@B*2 ink@Ml end_any@Br*2",
    "3/flat_top": "edge_t@T bay_L@M bay_L@B*2 edge_b@B*2 ink@Bl edge_r@Mr end_any@Tl end_any@Bl ?bay_L@T ?bay_R@T ?end_any@Tr ?end_any@Ml ?end_any@Mc ?mid_h@M ?ink@Br | loop*2 end_down@B*2 bay_R@M ink@Ml end_any@Br*2",
    "3/curled": "bay_L@T*2 bay_L@M edge_b@B*2 ink@Bl edge_r@Mr end_any@Tl ?loop ?bay_L@B ?end_any@Bl ?end_any@Tr ?end_any@Ml ?end_any@Mc ?mid_h@M ?ink@Br | hole_big*2 loop_two*2 bay_R@T bay_R@B bay_R@M end_down@B*2 ink@Ml end_any@Br*2",
    "4/closed": "loop*2 edge_r@T end_any@Bc ?mid_h@M ?end_any@Bl ?end_any@Br ?end_any@Tr ?end_any@Mr | loop_tall*2 loop_two*2",
    "4/open": "bay_U@T*2 end_any@Mr end_any@B run_h@M ?end_any@Tl ?end_any@Tc ?end_any@Tr ?end_any@Ml ?end_any@Bl ?end_any@Bc ?end_any@Br ?edge_l@T ?edge_r@B ?mid_h@M ?mid_v@M ?bay_R@M | loop*2",
    "5/std": "bay_R@T*2 bay_R@M bay_L@B*2 edge_l@T edge_t@T ink@Ml end_any@Bl ?end_any@Tr ?end_any@Tl ?end_any@Ml ?bay_L@M ?mid_h@M ?edge_b@B | loop*2 bay_L@T*2 edge_r@Mr*1.5",
    "5/curled": "bay_R@T*2 edge_t@T*2 edge_l@T loop_lo end_any@Tr ?loop ?bay_R@M ?bay_L@B ?end_any@Tl ?end_any@Ml ?mid_h@M ?edge_b@B ?end_any@Bl | hole_big*2 loop_two*2 bay_L@T*2 loop_hi*1.5",
    "6/open": "end_ur@T*2 bay_R@M*2 bay_R@T edge_l@M ?end_any@Bl ?end_any@Bc ?end_any@Br ?end_any@Mr ?end_any@Ml ?loop ?bay_U@B ?bay_L@B ?mid_h@M ?end_any@Tr | bay_L@T*2 bay_L@M*2 loop_hi*2 loop_two*2 hole_big*1.5",
    "6/std": "loop_lo*3 loop bay_R@T ?end_any@Tr ?mid_h@M | loop_hi*2 loop_two*2 hole_big*1.5",
    "7/plain": "edge_t@T*2 bay_L@M*2 bay_L@B end_any@T end_any@B end_down@B ?end_any@Tl ?end_any@Tr ?end_any@Bl ?end_any@Br ?bay_L@T ?edge_b@T ?mid_h@M ?edge_r@Bc | loop*2 edge_b@B run_h@B ink@Bl",
    "7/crossed": "edge_t@T*2 mid_h@M bay_L@T end_any@T end_any@B end_down@B ?end_any@Tl ?end_any@Tr ?end_any@Bl ?end_any@Br ?end_any@Ml ?bay_L@M ?bay_L@B ?edge_b@T ?edge_r@Bc | loop*2 edge_b@B*2 run_h@B*2 ink@Bl",
    "8/stacked": "loop_two*2 waist*2 loop run_v@Mc ink@Tr ink@Bl ?mid_h@M ?mid_h@c ?bay_R@T ?bay_R@M ?bay_L@T ?bay_L@M ?end_any@Tl ?end_any@Tr ?run_h@T ?edge_t@T ?edge_b@B | loop_tall hole_big end_any@Bc",
    "8/collapsed": "loop_hi*2 ink@Bl*1.5 edge_l@l ink@Tr crossing@M ?junction@M ?end_any@B ?end_any@Bl ?end_any@Bc ?mid_h@M ?run_v@T ?bay_L@M ?bay_R@M | loop_lo*2 hole_big*2 bay_L@B*1.5 junction@Mr",
    "9/straight": "loop_hi*3 loop end_any@B ?end_any@Br ?end_any@Bc ?mid_h@M ?edge_t@T | loop_lo*2 loop_two*2 hole_big ink@Bl*1.5",
    "9/curved": "loop_hi*3 loop end_any@B bay_L@B ?end_any@Bl ?end_any@Bc ?mid_h@M ?edge_t@T | loop_lo*2 loop_two*2 hole_big",
    # ---- uppercase ----------------------------------------------------------
    "A/std": "bay_D@B*2 loop*1.5 run_d1@A run_d2@A end_any@Bl end_any@Br ?loop_hi ?end_any@Tl ?end_any@Tr ?mid_h@M ?stem_l | edge_t@T stem_r",
    "B/std": "B_body*3 loop_two*3 loop stem_l run_v@l*1.5 edge_t@T edge_b@B ?junction@M ?crossing@M ?bay_L@T ?bay_L@M ?bay_L@B ?mid_h@M ?bay_D@B ?bay_R@M ?end_any@Tl ?end_any@Bl | loop_tall*0.5 hole_big tail_below_r*1.5 end_any@Br*1.5",
    "C/std": OPEN_R,
    "D/std": "D_body*3 loop_tall*2 loop stem_lh@T stem_lh@B ?stem_l ?end_any@Tl ?end_any@Bl ?loop_two ?hole_big ?junction@T ?junction@B ?bay_L@T ?bay_L@M | waist*0.5 desc_l@A*1.5 asc_l@A*1.5 bay_D@B*1.5",
    "D/stem": "loop_right*2 end_any@l stem_lh@T hx2@U hx2@D loop ?edge_b@B ?vx2@c ?stem_lh@B ?end_any@Tl ?end_any@Bl ?end_any@Tr ?end_any@Br ?D_body ?loop_tall ?hole_big ?loop_two ?asc_l@A ?desc_l@A ?bay_R@T ?bay_R@B ?bay_L@T ?bay_L@M | waist*0.5 mid_h@M bay_D@B*1.5 hx1@B*0.5 hx1@T*0.5",
    "E/std": ("?E_arms bay_R@T*2 bay_R@B*2 mid_h@M stem_l edge_t@T edge_b@B end_any@Tr end_any@Br end_any@Mr "
              "?end_any@Tl ?end_any@Bl ?bay_R@M | loop*2"),
    "F/std": ("bay_R@T*2 edge_t@T stem_l end_any@Tr end_any@Mr ?end_any@Bl ?end_any@Bc ?mid_h@M ?bay_R@M ?end_any@Tl "
              "| run_h@B*2 loop*2"),
    "F/slant": ("edge_t@T*2 run_h@T end_any@Tr end_any@Mr junction@M end_any@B ?end_any@Bl ?end_any@Bc ?end_any@Tl "
                "?mid_h@M ?bay_R@M ?bay_R@T | loop*2 run_h@B*2 bay_D@B*1.5 end_any@Tl end_any@Tc"),
    "G/std": "bay_R@T*2 bay_R@M edge_t@T edge_b@B end_any@Tr ?vx3@c ?junction@M ?mid_h@M ?end_any@Mr ?loop ?bay_R@B | waist*2",
    "G/no_bar": "bay_R@T*2 bay_R@M edge_t@T edge_b@B end_any@Tr ?vx3@c ?mid_h@M ?end_any@Mr ?loop ?bay_R@B | waist*2",
    "H/std": "bay_U@Tc*2 bay_D@Bc*2 hx2@U hx2@D junction@r edge_t@M edge_b@M ?stem_l ?stem_r end_any@Tr end_any@Bl ?end_any@Tl ?end_any@Br ?bay_U@T ?bay_D@B | loop*2 edge_t@T*1.5 run_d1@A*0.5 run_d2@A*0.5",
    "I/stick": STICK,
    "I/serif": ("run_v@c*2 vx1@c stem_l stem_r edge_t@T edge_b@B end_any@Tl end_any@Tr end_any@Bl end_any@Br ?narrow "
                "run_h@B ?bay_R@T ?bay_R@M ?bay_R@B ?bay_L@T ?bay_L@M ?bay_L@B | loop*2 vx3@c*2 run_d1@M*1.5 run_d2@M*1.5"),
    "J/hook": "edge_r@T*2 bay_L@Mc hx2@D end_any@T end_any@Bl ?end_left@l ?end_ul@l ?end_any@Tr ?end_any@Ml ?bay_U@B ?bay_L@M ?bay_L@B ?narrow | loop*2 vx3@c*1.5",
    "J/bar": "edge_r@T*2 bay_L@Mc hx2@D edge_t@T end_any@Tl end_any@Bl ?junction@T ?end_left@l ?end_ul@l ?end_any@Tr ?end_any@Ml ?bay_U@B ?bay_L@T ?bay_L@M ?bay_L@B | loop*2 vx3@c*1.5",
    "K/std": "bay_R@M*2 edge_l@l*2 run_v@l bay_U@T bay_D@B end_any@Tl end_any@Tr end_any@Br junction@l*1.5 ?stem_l ?junction@Ml ?end_any@Bl ?bay_R@T ?bay_R@B | loop*2 end_ul@T*0.75 stem_r*1.5",
    "K/arrow": "bay_R@M*2 edge_l@l*2 run_v@l bay_U@T bay_D@B end_any@Tl end_any@Tr end_any@Br junction@l*1.5 ?stem_l ?junction@Ml ?end_any@Bl ?bay_R@T ?bay_R@B | loop*2 end_ul@T*0.75 stem_r*1.5",
    "L/std": "stem_l*2 edge_t@B*2 edge_b@B end_any@Tl end_any@Br ?end_any@Bl ?narrow | loop*2",
    "M/std": EM,
    "M/short_v": EM.replace("run_d1@A run_d2@A", "stem_l stem_r run_d2@A"),
    "N/std": "bay_U@Tr*2 bay_D@Bl*2 run_d2@A stem_l stem_r end_any@Bl end_any@Tr ?end_any@Tl ?end_any@Br ?bay_U@T ?bay_D@B | loop*2 bay_U@Tl bay_D@Br",
    "O/std": ROUND.replace("waist*1.5", "waist*0.75").replace(" |", " ?end_any@Tr ?end_any@Tl ?end_any@Tc |"),
    "P/std": "loop_hi*3 loop loop_right stem_l edge_t@T end_any@B ?end_any@Bl ?end_any@Tl ?mid_h@M | loop_lo*2 loop_two*2 loop_tall",
    "Q/std": "loop_tall*2 loop hole_big end_any@Br*2 ?loop_two ?loop_hi ?mid_h@M ?crossing@B ?junction@B",
    "Q/crossed": "hole_big*2 loop crossing@B end_any@Br ?junction@B ?loop_two ?loop_tall ?mid_h@M ?end_any@Bc",
    "R/std": "loop_hi*2 loop bay_D@B bay_R@A hx2@B ?stem_l edge_t@T end_any@Bl end_any@Br ?mid_h@M ?bay_R@M ?end_any@Tl | loop_two*2",
    "S/std": SNAKE,
    "T/std": ("edge_t@T*2 edge_b@T*2 junction@T end_any@Tl end_any@Tr end_any@Bc ?end_any@Bl ?end_any@Br "
              "?bay_R@T ?bay_R@M ?bay_L@T ?bay_L@M | loop*2 run_h@B*1.5"),
    "T/slant": ("edge_t@T*2 run_h@T end_any@Tl end_any@Tr end_any@B ?junction@T ?end_any@Bl ?end_any@Br ?end_any@Bc "
                "?bay_L@M ?bay_R@M ?bay_L@B ?bay_R@B | loop*2 bay_U@T*2 run_h@B*1.5 bay_L@Mc*1.5 end_left@l"),
    "U/std": "bay_U@T*3 hx2@U hx2@M edge_b@B end_any@Tl end_any@Tr ?stem_l ?stem_r ?bay_U@B ?end_any@B | loop*2 hx3@U*1.5 junction@M",
    "V/std": VEE,
    "W/std": DOUBLE_V,
    "X/std": CROSS,
    "Y/std": "bay_U@T*2 junction@M hx1@B*1.5 edge_l@B edge_r@B end_any@Tl end_any@Tr end_any@Bc ?end_any@Bl ?end_any@Br ?mid_v@M ?junction@T | loop*2 bay_U@B*2 bay_U@M hx3@M*2 edge_l@Ml crossing@M*1.5",
    "Y/one_stroke": "bay_U@T*2 hx1@B end_any@Tl end_any@Tr end_any@Bl ?bay_L@M ?bay_L@B ?end_any@Bc | loop*2 bay_U@B*2 hx3@M*2 bay_U@M*1.5",
    "Z/std": ZED,
    "Z/crossed": ZED.replace(" | ", " mid_h@M ?end_any@Ml ?end_any@Mr | "),
    # ---- lowercase ----------------------------------------------------------
    "a/round": "a_body*3 loop*2 loop_left*1.5 vx1@r end_any@Br ax1@B edge_t@T ?hole@B ?hx2@B ?edge_r@B ?loop_hi ?loop_tall ?end_any@Tr ?end_any@Tc ?mid_h@M | loop_two*2 vx3@c*1.5 bay_R@c*1.5 end_any@l",
    "a/storey": "loop*2 loop_lo*2 vx1@r edge_t@T ?bay_L@T ?bay_L@M ?end_any@Tl ?end_any@Ml ?end_any@Br ?mid_h@M ?bay_L@B | loop_hi*2 stem_l*1.5",
    "a/closed": "loop_two*2 loop_lo*1.5 vx1@r edge_t@T ?end_any@Br ?edge_r@B ?junction@M ?mid_h@M ?end_any@Tl ?end_any@Ml | stem_l*1.5 loop_hi*1.5 tail_below_r*1.5",
    "b/std": "b_body*3 asc_l@A*2 loop_lo*3 loop ?edge_l@T ?end_any@Tl ?end_any@Tc ?loop_right ?end_any@Bl ?mid_h@M | loop_hi*2 loop_two*2 loop_tall*1.5 asc_r@A*0.5",
    "c/std": OPEN_R,
    "d/std": "d_body*3 asc_r@A*2 loop_lo*3 loop ?edge_r@T ?end_any@Tr ?end_any@Tc ?loop_left ?end_any@Br | loop_hi*2 loop_two*2 asc_l@A*0.5",
    "e/std": "e_body*2 loop_hi*2 loop bay_R@B*2 end_any@Br ?end_any@Tr ?end_any@Tl ?bay_R@M ?mid_h@M | loop_lo*2 stem_l*1.5 hole_big*1.5",
    "f/std": ("junction@M*1.5 bay_R@T end_any@Tr end_any@Bc ?crossing@M ?end_any@Bl ?end_any@Ml ?end_any@Mr ?mid_h@M ?bay_R@M ?end_any@Tc "
              "?bay_L@M ?bay_L@B ?narrow | loop*2 run_h@B*2"),
    "g/nine": "tail_below_r*2 loop_hi*3 loop bay_L@B end_any@Bl vx3@c ?end_any@Bc ?mid_h@M ?bay_L@M | loop_two loop_right*2",
    "g/loop": "loop_two*2 loop vx3@c ?loop_hi ?loop_tall ?mid_h@M | stem_l*2 loop_right*2 hole_big*2",
    "h/std": "bay_D@B*2 hx1@T*2 edge_l@T end_any@Tl end_any@Br ?stem_l ?end_any@Bl ?end_any@Bc ?bay_D@M | loop*2 ink@Tr*1.5 edge_t@T",
    "i/std": "gap*3 narrow end_any@Bc ?end_any@Tc ?end_any@Tl ?end_any@Tr | loop*2",
    "j/std": "gap*3 end_any@Bl ?bay_L@M ?bay_L@B ?end_any@Tc ?end_any@Bc ?end_any@Tr | loop*2",
    "j/nodot": "end_left@l*1.5 bay_L@B ax2@c end_any@T ?end_any@Tc ?gap ?end_any@Bl ?bay_L@M ?end_ul@l | loop*2 bay_D@A*1.5 vx2@c*1.5",
    "k/std": "bay_R@M*2 edge_l@l*2 run_v@l bay_D@B end_any@Tl end_any@Mr end_any@Br junction@l*1.5 ?stem_l ?junction@Ml ?end_any@Tr ?bay_U@T ?end_any@Bl | loop*2 end_ul@T*0.75 stem_r*1.5",
    "l/stick": STICK,
    "l/tail": "narrow edge_l@T end_any@Tc end_any@Br ?end_any@Tl ?end_any@Bc ?bay_R@B | loop*2",
    "m/std": "bay_D@B*2 mid_v@M ink@Tc ink@Tr end_any@Tl end_any@Bl end_any@Bc end_any@Br ?bay_D@T | loop*2 bay_U@T*2",
    "n/std": "bay_D@B*2 bay_D@M edge_t@T ink@Tc ink@Tr stem_l end_any@Tl end_any@Bl end_any@Br ?bay_D@T | loop*2 bay_U@T*2 mid_v@M",
    "o/std": ROUND.replace("waist*1.5", "waist*0.75").replace(" |", " ?end_any@Tr ?end_any@Tl ?end_any@Tc |"),
    "p/std": "tail_below_l*2 loop_hi*3 loop stem_l edge_t@T end_any@B ?end_any@Bl ?end_any@Tl ?mid_h@M | loop_lo*2 loop_two*2 loop_tall",
    "q/std": "tail_below_r*2 loop_hi*3 loop edge_r@B end_any@B ?end_any@Br ?end_any@Tr | loop_lo*2 loop_two*2 loop_right*2 bay_D@B*1.5",
    "q/hook": "tail_below_r*2 loop_hi*3 loop edge_r@B end_any@Br ?end_any@Tr | loop_lo*2 loop_two*2 loop_right*2 bay_D@B*1.5",
    "r/std": "edge_t@T edge_l@l end_any@Tr end_any@B vx1@c ?end_any@Tl ?end_any@Bl ?bay_R@M ?narrow | loop*2 bay_D@B*2 vx2@c*1.5 mid_h@M bay_U@T*1.5",
    "s/std": SNAKE,
    "t/std": ("bay_R@M bay_R@B junction@M ?crossing@M end_any@Tc end_any@Br ?end_any@Tl ?end_any@Tr ?end_any@Ml ?end_any@Mr "
              "?bay_L@T ?bay_L@M ?mid_h@M ?narrow | loop*2"),
    "t/cross": "junction@M*2 ?crossing@M end_any@Tc end_any@Bc end_any@Ml end_any@Mr ?end_any@Tl ?end_any@Tr ?end_any@Bl ?end_any@Br ?run_h@M ?run_v@c ?junction@M | loop*2",
    "q/straight": "tail_below_r*2 loop_hi*3 loop edge_r@B run_v@r end_down@B ?end_any@Br ?end_any@Bc ?end_any@Tr | loop_lo*2 loop_two*2 loop_right*2 bay_L@B*1.5 bay_D@B",
    "u/std": "bay_U@T*3 hx2@U hx2@M end_any@Tl end_any@Tr end_any@Br ?stem_l ?bay_U@B ?edge_b@B | loop*2 hx3@U*1.5",
    "v/std": VEE,
    "w/std": DOUBLE_V,
    "w/round": "bay_U@T*2 hx3@M*2 stem_l stem_r end_any@Tl end_any@Tr ?end_any@Tc ?bay_U@B ?mid_v@M | loop*2 bay_D@B",
    "x/std": CROSS,
    "y/std": "bay_U@T*2 junction@M end_any@Tl end_any@Tr end_any@B ?end_any@Bl ?end_any@Bc ?narrow ?bay_L@M ?bay_L@B | loop*2 bay_U@B*2 hx3@M*2",
    "y/u_tail": "bay_U@T*2 bay_L@B end_any@Tl end_any@Tr end_any@Bl ?end_any@Tc ?mid_h@M ?bay_U@B ?mid_v@M ?loop | hx3@M*2",
    "z/std": ZED,
}

# ---- referees ------------------------------------------------------------------
# "a>b": ("evidence for a (positive weights) and for b (negative weights)", bias, why)
REF_GATE = 1.5      # a referee is silent once the weaker of its pair is this far below the top score
REF_WEIGHT = 0.75    # a referee can move the pair apart by at most 2 x this
REFEREES = {
    # ---- digit pairs (studied on MNIST training images) ----
    '0>4': ('hole@B*1.5 ink@Bl hole_big*0.5 bay_U@T*-1.5 end_any@Bc*-1', 0, 'a 0 is closed down to the bottom; a 4 is open at the top and its stem ends at the bottom'),
    '0>5': ('edge_r@Mr*1.5 loop hx2@M edge_t@T*-1 bay_R@T*-1 end_any@Tr*-1', 0, "a 0's right side runs down through the middle around its loop; a 5 has a top bar ending at the top right above an opening to the right"),
    '0>6': ('edge_r@r*1.5 hole_big hx2@U*0.5 bay_R@A*-1.5 end_ur@A*-1', 0, "a 0's right side runs the whole way down; a 6 opens to the right above its bowl"),
    '0>8': ('hx2@M ink@Ml hole_big*0.5 edge_r@Mr*0.5 hx1@M*-1.5 vx3@c*-1 waist*-0.5 junction@Mc*-2 crossing@M*-1.5', 0.5, "a 0 is one oval whose middle rows cross both sides; an 8's middle rows cross only its waist, its strokes meet or cross in the centre, and its centre columns cross three strokes (where a 0's pen start overlaps, it can show a small junction, so the crossing counts for more)"),
    '9>0': ('loop_hi*1.5 hx1@B end_any@B edge_l@l*-1 ink@Bl*-1 loop_lo*-1', 0, "a 9's loop sits high and its bottom rows cross one stroke, which ends at the bottom; a 0's loop reaches the bottom and its left side runs down the left"),
    '2>1': ('vx2@A*1.5 edge_b@B bay_L@A end_up@T*-0.5 narrow*-0.5', -0.5, 'a 2 has a base and a top curve, so its columns cross two strokes; a 1 is a single stroke'),
    '3>1': ('bay_L@A*2 vx3@c vx1@c*-1', -0.5, "a 3 opens to the left and its columns cross three strokes; a 1's columns cross one"),
    '4>1': ('hx2@A*2 end_any@M*0.5 narrow*-0.5', -0.5, "a 4's rows cross two strokes (arm and stem); a 1's rows cross one"),
    '1>6': ('line narrow hx2@B*-2 loop*-1 bay_R@M*-1', 0.5, "a 1 is one straight stroke near the centre; a 6's bowl makes its bottom rows cross two strokes"),
    '7>1': ('edge_t@T*1.5 vx2@c run_h@T*0.5 marg_b@l end_up@T*-1.5 narrow*-0.5 end_dl@B*-0.5', -0.3, "a 7 has a top bar with empty space under its left end; a 1's top ends pointing up"),
    '3>2': ('bay_L@B end_ul@B*1.5 vx3@c end_any@Ml*0.5 end_any@Mc*0.5 edge_r@Mc*0.5 end_any@Br*-1.5 end_dr@B*-0.5 end_dl@B*-1 vx2@c*-0.5 edge_b@M*-1 hole@Bc*-1 junction@B*-1', 0, "a 3's bottom tail ends pointing up-left, it has a cusp in the middle, and its centre columns cross three strokes; a 2's base ends at the bottom right, its pen start points down-left, its diagonal has nothing under it, and a looped base makes a junction at the bottom"),
    '5>2': ('edge_t@Tr*1.5 end_any@Tr end_ul@B bay_R@T bay_L@T*-1.5 end_any@Tl*-1 end_any@Br*-1 edge_b@M*-0.5 run_d1@M*-0.5', 0, "a 5's top bar runs to the top right and its tail curls up-left; a 2 starts at the top left, opens to the left under its top curve, and ends at the bottom right"),
    '6>2': ('end_ur@A*1.5 loop_lo bay_R@T bay_L@A*-2 edge_b@B*-0.5 end_any@Br*-1', 0, "a 6's stem rises to the top right above an opening to the right, and its bowl is closed; a 2 opens to the left, has a base, and ends at the bottom right"),
    '7>2': ('hx1@B edge_l@B edge_t@T*0.5 end_down@B run_h@B*-1.5 hx2@D*-1 end_any@Br*-0.5 ink@Bl*-0.5', -0.3, "a 7's bottom is one stroke with nothing to its left; a 2 has a base, so its lower rows cross two strokes"),
    '2>8': ('edge_b@B end_any@Br*1.5 bay_L@M*1.5 crossing@M*-2 hole@T*-1 hole@B*-0.5 run_v@Bc*-0.5', 0, 'a 2 ends at the bottom (right), its diagonal has nothing under it and it opens to the left under its top curve; an 8 crosses in the middle and has loops'),
    '9>2': ('loop_hi*1.5 ink@Ml*0.5 bay_L@M*-1.5 edge_b@B*-1 run_h@B*-0.5', 0, 'a 9 has a closed head; a 2 opens to the left in the middle and has a flat base'),
    '5>3': ('bay_R@T*1.5 end_right@T ink@Ml vx2@c*0.5 bay_L@T*-1.5 end_any@Mc*-1 edge_r@r*-0.5 junction@Mr*-1 junction@Mc*-0.5', 0, 'a 5 opens to the right at the top and has a stem on the left; a 3 opens to the left at the top and its two arms meet in the middle (a cusp or a junction on the right)'),
    '3>7': ('vx3@c*2 edge_b@B mid_h@c ink@Br marg_r@T*0.5 end_down@B*-1.5 vx1@r*-1 ink@Mr*-0.5', -0.5, "a 3 has three arms stacked in the centre and a lower bowl reaching the bottom right; a 7's bar reaches the top right and its stem crosses the middle right to end pointing down"),
    '3>8': ('bay_L@A*1.5 end_any@Ml crossing@M*-2 loop*-1', 0, "a 3 opens to the left and has a cusp in the middle; an 8's strokes cross in the middle around closed loops"),
    '6>4': ('hx1@T loop_lo*1.5 bay_R@T*0.5 bay_R@c*0.5 hx2@T*-1.5 bay_U@T*-1.5 hx1@B*-0.5 crossing@M*-1.5 end_any@B*-1', 0, "a 6's top rows cross one rising stroke, it opens to the right above a loop at the bottom; a 4's top rows cross its two arms (or it opens at the top), its bar crosses the stem, and its stem ends at the bottom"),
    '4>7': ('bay_U@A*1.5 junction@M hx2@T run_h@M*0.5 edge_t@T*-1.5 bay_L@M*-1 bay_D@M*-1.5', 0, 'a 4 opens at the top, its tips point up and its bar meets the stem; a 7 has a top bar with an opening under it, to the left of its stem'),
    '8>4': ('hole@B*1.5 loop bay_U@T*-1.5 vx1@c*-1', 0, 'an 8 has a loop at the bottom; a 4 is open at the top and its centre columns cross one stroke'),
    '9>4': ('loop_hi*1.5 loop*0.5 edge_t@T*0.5 run_h@T*0.5 mid_h@c*0.5 bay_U@A*-1.5 end_up@T*-1 end_ur@T*-0.5 end_any@M*-1 edge_t@M*-1 crossing@M*-1.5', 0, "a 9's head is closed with a curved top, and its only end is its tail; a 4 opens at the top, its tips point up, its bar has nothing above it, ends in the middle, and crosses the stem"),
    '5>6': ('edge_t@T*1.5 ink@Tr run_h@T*0.5 bay_L@A*0.5 end_right@T*0.5 end_any@Ml*0.5 end_ur@r*-1.5 hole@Mr*-1 end_any@Mr*-0.5 junction@B*-1', -0.2, "a 5 has a flat top bar reaching the top right (even when its bowl curls shut); a 6's stem rises to an end pointing up-right and joins its bowl low down"),
    '5>7': ('bay_R@A*1.5 edge_b@A vx3@c*0.5 end_down@B*-1.5 ink@Mr*-1 end_any@Bc*-0.5', 0, "a 5 opens to the right and has a bottom curve; a 7's stem crosses the middle right and ends pointing down"),
    '8>5': ('junction@Mc*2 crossing@M loop*0.5 edge_t@T*-1 bay_R@T*-1 end_any@Tr*-1', -0.5, "an 8's strokes meet in the centre; a 5 has a top bar ending at the top right above an opening to the right"),
    '8>6': ('hole@T*1.5 hx2@T run_v@Bc*0.5 crossing@M*1.5 bay_R@T*-1.5 hole@Mr*-0.5 hx1@T*-0.5 junction@l*-0.5', 0, "an 8 has a loop at the top and its strokes cross in the middle; a 6's top is a single stroke with an opening to the right, and its stem joins the bowl on the left"),
    '9>7': ('loop*1.5 mid_h@A*0.5 vx3@c*0.5 bay_L@M*-1 end_any@Tl*-0.5', -0.3, 'a 9 has a head (closed, or a bar under it); a 7 opens to the left under its bar'),
    '8>9': ('hole@B*1.5 hx2@B waist*0.5 ink@Bl ink@Tr*0.5 edge_l@l end_any@B*-1 bay_L@B*-1 hx1@B*-0.5 junction@Mr*-1', 0, "an 8's lower half reaches the bottom left (a loop, even a squashed one) and it has a left side; a 9 opens to the left under its head, its tail joins the head on the right and ends at the bottom"),
    # ---- digit / letter look-alikes (5/S, 9/q, 0/O and 1/I have no referee: no reliable difference) ----
    '0>B': ('hole_big junction@M*-1.5 stem_l*-1 waist*-1', 0, "a 0 is one big hole; a B's two bowls meet the middle of a straight left stem"),
    '0>d': ('hole@Mr*1.5 hole_big edge_l@l loop_left*-1.5 edge_r@Mc*-1', 0, "a 0 is one oval: its hole reaches into the right half and its left side runs down the left; a d's bowl sits low on the left of a stem rising on the right"),
    'L>1': ('edge_b@B*1.5 run_h@B end_any@Br*0.5', 0, 'an L has a foot running right along the bottom; a 1 (or l) is one stroke'),
    'T>1': ('junction@A*1.5 edge_t@T', 0, "a T's bar meets its stem; a 1 is one stroke"),
    't>1': ('junction@A*1.5 edge_t@M*0.5', 0, "a t's bar crosses its stem; a 1 is one stroke"),
    '2>a': ('run_d1@M bay_L@M*1.5 edge_b@B*0.5 edge_r@Mr*-1 run_v@r*-1', 0, "a 2's diagonal runs down to the left through the middle, open to its left; an a has a straight stem down its right side"),
    '3>J': ('vx3@c*1.5 end_dl@M*0.5 vx1@c*-1', 0, "a 3's centre columns cross three arms (top, middle, bottom); a J's cross one or two"),
    '3>Z': ('bay_L@B*1.5 run_d1@M*-1 end_dl@Bl*-0.5', 0, "a 3's lower bowl opens to the left; a Z's diagonal runs down to the left through the middle to its base"),
    'r>4': ('end_any@Mr*1.5 edge_b@r bx2@M*-1.5', 0, "an r's arm ends at the middle right with nothing under it; a 4's diagonal lines through the middle cross arm and stem"),
    '4>U': ('hx1@B*1.5 edge_r@B hx2@D*-1 ink@Bl*-1 run_v@l*-1 edge_b@B*-1', 0, "a 4's bottom is a single stem on the right; a U's sides come down to a round bottom across its width"),
    '4>u': ('hx1@B*1.5 edge_r@B hx2@D*-1 ink@Bl*-1', 0, "a 4's bottom is a single stem on the right; a u's sides come down to a round bottom across its width"),
    'V>4': ('edge_l@Ml*1.5 bx1@B vx2@A*-1.5', 0, "a V's left arm runs down the middle left to the point; a 4's columns cross its bar and stem"),
    '4>W': ('run_h@M crossing@M hx3@M*-1.5 hx3@U*-1', 0, "a 4's bar crosses its stem in the middle; a W's middle rows cross three or more strokes"),
    'X>4': ('end_ul@T*1.5 bx1@B', 0, "an X's upper left stroke ends pointing up-left; a 4's arm stands up from its bar"),
    '4>y': ('run_h@M junction@r ink@Mr*0.5 hx1@M*-1 end_any@Bl*-0.5', 0, "a 4's bar runs across the middle and meets the stem on the right; a y's middle rows cross one stroke and its tail ends at the bottom left"),
    'Y>4': ('ax1@T end_any@M*-1.5', 0, "a Y's arms end at the top; a 4's bar ends in the middle"),
    'G>5': ('hx2@M*1.5 end_any@l*-1 bx1@l*-0.5', 0, "a G's middle rows cross its left side and its bar; a 5's stem starts at a stroke end on the left"),
    '5>T': ('edge_t@Tr hx2@D junction@c*-1.5 junction@Tc*-1', 0, "a 5's bar ends at the top right and its bowl below crosses two strokes; a T's bar crosses a stem in the centre"),
    '6>b': ('junction@B mid_v@M run_v@l*-1', 0, "a 6's stem curls into its bowl and joins it low; a b's stem runs straight down the left"),
    'C>6': ('bay_R@Mc loop*-1.5', 0, 'a C stays open to the right; a 6 closes a loop'),
    'd>6': ('hole@l*1.5 bay_R@T*-1.5', 0, "a d's bowl sits on the left under a stem rising on the right; a 6's stem rises on the left and opens to the right at the top"),
    'G>6': ('ax2@T*1.5 edge_l@c*-1 ax1@T*-1', 0, "a G's (or g's) top is a curve the / lines cross twice; a 6's top is one rising stroke"),
    'g>6': ('ax2@T*1.5 edge_l@c*-1 ax1@T*-1', 0, "a g's head sits on top, so the / lines there cross it twice; a 6's top is one rising stroke"),
    '7>S': ('run_d1@M*1.5 bay_R@A*-1.5 vx3@c*-1', 0, "a 7's stem runs down to the left through the middle; an S opens to the right and its centre columns cross three strokes"),
    '7>T': ('edge_r@r*1.5 ink@Tl*0.5 junction@c*-1.5 junction@T*-1', 0, "a 7's bar ends at the top right, where its stem starts down the right side; a T's bar crosses the top of a stem in the centre"),
    '7>Y': ('edge_t@T*1.5 run_h@T bay_U@T*-1.5', 0, 'a 7 has a flat bar across its top; a Y opens upward between its arms'),
    '8>B': ('crossing@M*1.5 run_v@B*0.5 marg_l@B*0.5 stem_l*-1 run_v@l*-1', 0, "an 8's strokes cross in the centre and its sides curve; a B's two bowls hang off a straight stem down the left"),
    '8>O': ('junction@M*1.5 mid_h@c vx3@c waist*0.5 vx2@c*-1', 0, "an 8's strokes meet in the centre, so its centre columns cross three strokes; an O's cross two"),
    'P>8': ('vx1@c hole@B*-1.5 hx2@B*-1', 0, "a P's lower half is its stem alone; an 8 closes a second loop at the bottom"),
    '9>P': ('vx3@c end_dl@B*0.5 edge_r@r loop_right*-1.5 edge_l@l*-1', 0, "a 9's tail drops from the right of its head; a P's head sits to the right of a stem down the left"),
    # ---- letter pairs, by the first letter of the pair ----
    # A
    'B>A': ('run_v@l*1.5 loop_two bay_D@B*-1.5', 0, "a B has two bowls on a straight left stem; an A's legs open downward"),
    'a>B': ('end_any@Br*1.5 ax1@r hole@r*-1.5 run_v@l*-1.5 stem_l*-1 vx1@r', 0, "an a's tail ends at the bottom right and its right side is one stroke; a B's bowls bulge to the right of a straight stem on the left"),
    'a>b': ('loop_left*1.5 loop_right*-1.5 edge_l@l*-1', 0, "an a's bowl sits left of its stem; a b's sits right of a stem on the left"),
    'b>A': ('asc_l@A*1.5 loop_lo bay_D@B*-1.5', 0, "a b's stem runs straight down the left; an A's legs open downward"),
    'c>a': ('bay_R@Mc*1.5 vx1@r*-1 a_body*-1', 0, 'a c stays open to the right; an a is closed on the right by its stem'),
    'a>D': ('loop_left*1.5 end_any@l*-1.5 stem_lh@T*-1.5 stem_lh@B*-1 a_body*1.5', 0, "an a's bowl sits left of its stem; a D's straight stem is on the left beside its bowl from top to bottom and leaves stroke ends there"),
    'a>d': ('edge_t@T*1.5 run_h@T ink@Tl run_v@Tr*-1.5 end_up@T*-1', -0.25, "an a's bowl reaches the top left with a flat top edge; a d's stem rises alone at the top right and ends pointing up"),
    'e>A': ('run_h@B*1.5 bay_D@B*-1.5', 0, "an e's bottom is a curve along the base; an A's legs open downward"),
    'e>a': ('desc_l@M*1.5 hx1@M loop_left*-1.5 hole@Ml*-1 vx1@r*-1.5 edge_r@B*-1 bay_R@B*1.5', 0, "an e's left curve continues below its loop and it opens to the right at the bottom; an a's bowl sits left of a straight stem that closes it on the right"),
    'f>A': ('bay_D@A*-2 hx2@D*-1', 0.5, "an f's bottom is its stem alone; an A's legs open downward"),
    'a>G': ('bay_R@c*-1.5 vx3@c*-1.5 ax1@B edge_b@Mc', 0, "an a's bowl bottom sits in the middle rows and the / lines at the bottom cross it once; a G opens to the right and its centre columns cross three strokes"),
    'a>g': ('end_any@Br*1.5 edge_b@Mc vx1@r tail_below_r*-1.5 end_down@B*-1.5', 0, "an a's stem ends in a tail at the bottom right and its bowl's bottom sits in the middle rows; a g's tail hangs below its head and ends pointing down"),
    'g>A': ('hole@l bay_D@B*-2 vx1@l*-1', 0, "a g closes its head; an A's legs open downward"),
    'H>A': ('loop*-2 stem_rh@M*-1', 0.5, "an H has no loop; an A's crossbar closes one"),
    'h>A': ('loop*-2 hole@M*-1', 0.5, "an h has no loop; an A's crossbar closes one"),
    'J>a': ('end_left@l*1.5 hx1@B loop*-2', 0, "a J's hook ends pointing left and its bottom rows cross one stroke; an a closes a bowl"),
    'K>A': ('bay_R@M loop*-1.5 edge_r@Mr*-1', 0, 'a K opens to the right between its arms; an A closes a loop'),
    'M>A': ('hx3@M*1.5 loop*-1.5', 0, "an M's middle rows cross three or four strokes; an A closes a loop"),
    'm>A': ('hx3@M*1.5 loop*-1.5', 0, "an m's middle rows cross three legs; an A closes a loop"),
    'O>A': ('hole_big*1.5 bay_D@B*-1.5', 0, "an O is one big bowl; an A's legs open downward"),
    'O>a': ('hole_big*1.5 end_any@Br*-1.5 loop_left*-1 vx1@r*-1', 0, "an O is one big bowl with no stroke ends; an a's bowl sits left of a stem that ends in a tail at the bottom right"),
    'o>a': ('loop_lo*1.5 hole@Mr junction@M*-2', 0, "an o is one closed curve, its hole reaching the right side; an a's bowl meets its stem in a junction"),
    'Q>A': ('edge_r@B bay_D@B*-1.5 edge_t@A*-1', 0, "a Q's bowl is closed below; an A's legs open downward"),
    'a>Q': ('loop_left*1.5 vx3@c*-1.5 hx3@A*-1.5 hole_big*-1 a_body*1.5 vx1@r', -0.5, "an a's bowl sits left of a straight stem; a Q is one big round bowl whose tail crosses it"),
    'a>q': ('edge_b@M*1.5 vx1@r hole@B marg_r@B*-1 hole@r*-0.5 end_down@B*-1.5 tail_below_r*-1', 0, "an a's bowl sits on the base beside its stem and reaches the bottom rows; a q's stem continues below its bowl, ends pointing down and leaves space to its right"),
    'q>A': ('ink@Bc bay_D@A*-1.5 vx1@l*-1', 0, "a q's stem stands alone at the bottom; an A's legs open downward"),
    'A>R': ('junction@r*1.5 bay_R@A*-2', 0, "an A's crossbar meets its right leg in a junction; an R opens to the right between its bowl and its leg"),
    'a>R': ('a_body*1.5 vx1@r bay_R@A*-2', 0, "an a's bowl sits left of a straight stem on the right, which its right columns cross once; an R opens to the right between its bowl and its leg"),
    'a>y': ('loop*1.5 end_any@Br bay_U@T*-1.5 end_any@Bl*-1', 0, 'an a closes its bowl and its stem ends at the bottom right; a y opens at the top and its tail ends at the bottom left'),
    'a>z': ('loop*1.5 run_d1@M*-1.5', 0, 'an a closes a bowl; a z runs a long diagonal through its middle'),
    # B
    'B>D': ('junction@M*1.5 vx3@c mid_h@M hole_big*-1', 0, "a B's bowls meet its stem in the middle, so its centre columns cross three strokes; a D is one big bowl"),
    'b>D': ('asc_l@A*1.5 loop_lo loop_tall*-1.5 hx2@U*-1.5 ax2@T*-1 asc_l@M*1.5', 0, "a b's stem continues above its low bowl; a D's bowl reaches the top, so its upper rows cross stem and bowl"),
    'd>b': ('hole@l*1.5 hole@r*-1.5 asc_l@A*-2 b_body*-1 loop_right*-1 asc_r@A*2 d_body loop_left', 0, "a d's bowl sits left of its stem, which continues above it; a b's bowl sits right of its stem, which continues above it"),
    'e>B': ('bay_R@B*1.5 stem_l*-1 loop_lo*-1', 0, 'an e opens to the right at the bottom; a B closes a bowl there on a straight stem'),
    'B>g': ('run_v@l*1.5 loop_two tail_below_r*-1.5 desc_r@A*-1.5 edge_r@Mc*-1 hole@r*1.5', 0, "a B's bowls sit right of a straight stem on the left; a g's stem continues below its head on the right"),
    'G>B': ('hole@r*-1.5 stem_lh@T*-1.5', 0, "a G has no bowl on the right; a B's bowls sit right of a straight stem"),
    'G>b': ('mid_h@A*1.5 ax1@T*-1 edge_l@Mc*-1.5', 0, "a G's bar runs into its middle; a b's stem runs straight down its left with nothing crossing it"),
    'b>g': ('asc_l@A*1.5 tail_below_r*-1.5', 0, "a b's stem rises above its bowl; a g's tail hangs below its head"),
    'b>h': ('loop*1.5 loop_lo bay_D@B*-1.5', 0, "a b closes its bowl; an h's leg leaves an opening downward"),
    'B>o': ('junction@M mid_h@c vx2@c*-1.5 vx3@c*2 mid_h@A', 0, "a B's bowls meet its stem in the middle, so its centre columns cross three strokes; an o's cross two"),
    'O>B': ('vx2@c*1.5 vx3@c*-2 mid_h@A*-1', 0, "an O's centre columns cross its top and bottom only; a B's waist makes them cross three times"),
    'O>b': ('edge_t@T*1.5 ink@Tr edge_r@Mr', 0, "an O's top runs across the top to its right side; a b's top is only its stem, with nothing at the top right"),
    'o>b': ('edge_t@T*1.5 ink@Tr edge_r@Mr', 0, "an O's top runs across the top to its right side; a b's top is only its stem, with nothing at the top right"),
    'Q>B': ('hx3@A*1.5 junction@B ax2@B*-1.5 run_v@l*-1.5 waist*-1 hole_big*1.5 end_any@Br', 0, "a Q's tail crosses its big bowl low down, so rows there cross three strokes; a B has a straight stem on the left and a waist between two bowls"),
    # C
    'C>E': ('mid_h@M*-2 junction@M*-1.5 E_arms*-1.5 end_any@M*-1', 0.5, "a C has nothing across its middle; an E's middle arm joins its stem, with openings to the right above and below it"),
    'c>E': ('mid_h@M*-2 junction@M*-1.5', 0.5, "a c has nothing across its middle; an E's middle bar joins its stem"),
    'e>C': ('loop_hi*1.5 mid_h@M', 0, 'an e closes its top into a loop with a bar; a C stays open'),
    'e>c': ('loop_hi*1.5 mid_h@c', 0, 'an e closes a loop with its bar; a c stays open'),
    'F>c': ('run_h@B*-2', 0.25, 'an F has no base; a c curves round into a base along the bottom'),
    'f>C': ('hx1@B run_h@B*-1.5 ink@Ml*-1', 0, "an f's stem stands alone at the bottom; a C curves round the left into a base"),
    'f>c': ('junction@M*1.5 run_h@B*-1', 0, "an f's bar crosses its stem; a c is one curve"),
    'C>G': ('bay_R@Mc vx3@c*-1.5 junction@M*-1', -0.25, "a C opens wide to the right in the middle and its centre columns cross two strokes; a G's bar closes part of that opening, so its centre columns cross three"),
    'c>G': ('bay_R@Mc vx3@c*-1.5 junction@M*-1 hx2@D*-1.5 bay_R@Tc*-1 hx1@D ax1@B', -0.25, "a c opens wide to the right in the middle and its lower rows cross one curve; a G's bar closes part of that opening, so its centre columns cross three strokes and its lower rows two"),
    'C>k': ('bay_R@Mc junction@M*-2 junction@l*-1', 0.5, "a C is one curve with no joins; a k's arms meet its stem"),
    'L>C': ('end_any@l*1.5 ax1@T ax2@A*-2', -0.5, "an L's stem is crossed once by every / line and starts at a stroke end on the left; a C's curve is crossed twice"),
    'L>c': ('end_any@l*1.5 ax1@T ax2@A*-2', -0.5, "an L's stem is crossed once by every / line and starts at a stroke end on the left; a C's curve is crossed twice"),
    'c>R': ('bay_R@Mc*1.5 bay_D@B*-1 stem_l*-1', 0, "a c opens to the right in the middle; an R's legs open downward from a straight stem"),
    'S>c': ('vx3@c*1.5 S_body bay_L@B bx2@r*-1.5', 0, "an S's spine slants across its middle and its lower curve opens to the left; a c's \\\\ lines cross its curve twice"),
    's>c': ('vx3@c*1.5 S_body bay_L@B bx2@r*-1.5', 0, "an s's spine slants across its middle; a c's \\\\ lines cross its curve twice"),
    'C>Z': ('edge_l@l bay_L@M*-1.5', 0, "a C's left side runs down the left; a Z opens to the left under its top bar"),
    'C>z': ('bay_R@Mc*1.5 edge_l@l run_d1@M*-1.5 bay_L@M*-1', 0, 'a C is one curve open to the right; a z runs a long straight diagonal through its middle and opens to the left under its top'),
    'c>z': ('bay_R@Mc*1.5 edge_l@l run_d1@M*-1.5 bay_L@M*-1', 0, 'a c is one curve open to the right; a z runs a long diagonal through its middle'),
    # D
    'G>d': ('vx3@A*1.5 bx1@c*-1.5 vx1@c*-1', 0, "a G's centre columns cross three strokes (top, bar, bottom); a d's centre is crossed once"),
    'J>d': ('bay_L@Mc end_any@Ml loop*-1.5', 0, "a J's hook stays open; a d closes a bowl"),
    'D>O': ('stem_lh@T*1.5 end_any@l end_ul@T run_d1@A*-1.5 ink@Tr*-1', 0, "a D has a straight stem beside its bowl at the top and leaves stroke ends on the left, often one pointing up-left; an O's sides run on the slant and it fills its top right"),
    'D>o': ('stem_lh@T*1.5 end_any@l end_ul@T run_d1@A*-1.5 ink@Tr*-1 run_v@Ml*1.5 edge_l@Mc*-1', -0.25, "a D has a straight stem beside its bowl at the top and down the middle left, and leaves stroke ends on the left; an o's sides run on the slant and it fills its top right"),
    'O>d': ('edge_l@l*1.5 loop_left*-1.5 bx1@T*-1', 0, "an O's left side is a curve down the whole left; a d's bowl sits left of a stem that rises alone at the top"),
    'o>d': ('edge_l@l*1.5 loop_left*-1.5 bx1@T*-1', 0, "an O's left side is a curve down the whole left; a d's bowl sits left of a stem that rises alone at the top"),
    'D>P': ('edge_b@B*1.5 hx2@D hx1@B*-2', -0.5, "a D's bowl closes along a bottom edge, so its lower rows cross stem and bowl; a P's bottom is its stem alone"),
    'D>p': ('edge_b@B*1.5 hx2@D hx1@B*-2', -0.5, "a D's bowl closes along a bottom edge, so its lower rows cross stem and bowl; a p's bottom is its stem alone"),
    'Q>D': ('end_any@Br*1.5 crossing@B edge_l@l*-1 end_any@l*-1', 0, "a Q's tail crosses out at the bottom right; a D's straight stem runs down the left"),
    'D>R': ('loop_tall*1.5 D_body bay_D@B*-1.5', 0, "a D is closed down to the bottom; an R's legs open downward"),
    # E
    'F>E': ('edge_b@M*1.5 vx3@c*-1.5 bay_R@B*-1 mid_h@c*-1', 0, "an F's middle arm has nothing below it and its centre columns cross two bars; an E has a third arm along the bottom"),
    'E>G': ('hx1@D*1.5 hx1@M edge_t@T hx2@D*-1.5 ink@Mr*-1', 0, "an E's middle and lower rows cross only its spine (its arms are horizontal) and its top is a flat bar; a G's lower rows cross its left side and the right side curling up"),
    'e>G': ('loop_hi*1.5 bay_R@B vx3@c*-1', 0, "an e closes a small loop at the top and opens to the right below it; a G's bar crosses its centre columns a third time"),
    'e>g': ('run_h@B*1.5 mid_h@c tail_below_r*-1.5 end_down@B*-1', 0, "an e's bottom is a curve along the base; a g's tail hangs below its head"),
    'K>E': ('hx2@A*1.5 vx3@c*-1.5', 0, "a K's rows cross stem and arm; an E's arms run straight across, so its centre columns cross three bars"),
    'k>E': ('hx2@A*1.5 vx3@c*-1.5', 0, "a k's rows cross stem and arm; an E's arms run straight across, so its centre columns cross three bars"),
    'e>p': ('run_h@B*1.5 vx3@c mid_h@c end_down@B*-1.5 end_any@Bl*-1', 0, "an e's bottom is a horizontal curve and its centre columns cross loop, bar and bottom; a p's stem ends pointing down at the bottom left"),
    'Q>e': ('hole@Ml*1.5 junction@B hx1@M*-1.5 bay_R@c*-1', 0, 'a Q is closed on both sides; an e opens to the right under its loop'),
    'q>e': ('edge_l@Bc*1.5 narrow wide_r*-1', 0, "a q's stem drops straight from the bowl; an e opens to the right under its loop"),
    'R>e': ('bay_D@A*1.5 end_any@Bl hx2@B end_ur@M*-1.5 end_any@Mr*-1 run_h@Bc*-1', 0, "an R's legs open downward from its stem, which ends at the bottom left; an e's tail ends pointing up-right at the middle right"),
    'S>E': ('S_body*1.5 bay_R@B*-1.5', 0, 'an S is one curve with a spine slanting across its middle; an E opens to the right at the bottom'),
    's>E': ('S_body*1.5 bay_R@B*-1.5 junction@M*-1.5', 0, 'an s is one curve with a spine slanting across its middle; an E opens to the right at the bottom and its arms meet its stem'),
    't>e': ('loop*-2 hole@T*-1', 0.25, 'a t has no loop; an e closes one at the top'),
    'E>z': ('bay_L@A*-2 end_left@A*-1', 0.5, 'an E never opens to the left; a z opens to the left under its top bar'),
    # F
    'I>f': ('ax1@A*1.5 vx2@c*-1.5 junction@A*-1.5', 0, "an I (or i) is one upright stroke, crossed once by every / line; an f's crossbar crosses its stem, so its centre columns cross two strokes"),
    'i>f': ('ax1@A*1.5 vx2@c*-1.5 junction@A*-1.5', 0, "an I (or i) is one upright stroke, crossed once by every / line; an f's crossbar crosses its stem, so its centre columns cross two strokes"),
    'F>K': ('edge_t@T*1.5 run_h@T bay_R@T bay_R@B*-1.5 end_any@Br*-1', 0, "an F's top is a bar with an opening under it; a K's leg runs down to the bottom right"),
    'f>K': ('bay_R@T junction@M bay_R@B*-1.5 end_any@Br*-1', 0, "an f hooks over at the top and its bar crosses the stem; a K's leg runs down to the bottom right"),
    'f>k': ('hx1@B*1.5 ax1@A*-1', 0, "an f's bottom rows cross only its stem; a k's cross stem and leg"),
    'F>p': ('end_any@r', 0, "an F's arms end on the right; a p closes a bowl there"),
    'f>p': ('hole@r*-1.5 loop_right*-1 stem_rh@M*-1', 0.5, 'an f has no bowl; a p closes a bowl to the right of its stem'),
    'r>F': ('end_dr@M*1.5 vx2@A*-1.5', 0, "an r's arm ends pointing down-right; an F's centre columns cross two bars"),
    'T>F': ('ax1@T end_left@T bay_R@A*-1.5 vx2@r*-1.5', 0, 'a T is crossed once by the / lines at the top and its bar ends pointing left; an F opens to the right between its arms, so its right columns cross two'),
    't>F': ('edge_t@Mc*1.5 bay_R@M*-1.5 bay_R@c*-1', 0, "a t's bar has a top edge in the middle; an F opens to the right between its arms"),
    't>f': ('edge_t@Mc*1.5 end_dr@T*-1.5 end_any@Tr*-0.5 bay_R@Mc*-1.5', 0, "a t's bar has a top edge in the middle; an f's hook ends pointing down-right at the top and leaves an opening to the right under it"),
    'F>Y': ('edge_t@T bay_U@A*-2', 0, 'an F has a top bar and does not open upward; a Y opens upward between its arms'),
    'F>y': ('bay_U@A*-2 hx2@U*-1', 0.5, 'an F does not open upward; a y opens upward between its arms'),
    'f>Y': ('edge_t@T bay_U@A*-2', 0, 'an f has a top hook and does not open upward; a Y opens upward between its arms'),
    'F>Z': ('hx1@B bay_L@M*-1.5 ax1@A*-1', 0, "an F's bottom is its stem alone; a Z opens to the left under its top bar"),
    'F>z': ('edge_l@l bay_L@M*-1.5', 0, "an F's stem runs down the left; a z opens to the left under its top bar"),
    'f>Z': ('hx1@B bay_L@M*-1.5 ax1@A*-1', 0, "an f's bottom is its stem alone; a Z opens to the left under its top bar"),
    # G
    'g>p': ('hole@l bay_L@B edge_l@l*-1', 0, "a g's head sits left of its tail, which curls back left; a p's stem runs down the left"),
    'Q>g': ('hole_big junction@B bay_L@A*-1.5 desc_r@M*-1.5', 0, "a Q's tail stays at the bottom of its bowl; a g's stem continues below its head and curls left"),
    'g>q': ('bay_L@M*1.5 bay_L@B bay_L@c edge_r@Bc*-1.5 run_v@Bc*-1', 0, "a g's tail curls back under its head, so it opens to the left in the middle; a q's stem drops straight at the bottom centre"),
    'S>G': ('end_any@l*1.5 bx1@B S_body', 0, "an S's lower curve ends on the left and its spine slants across the middle; a G curls back into its bar"),
    'S>g': ('run_d2@M*1.5 end_any@Bl loop*-1.5', 0, "an S's spine runs down to the right and its lower curve ends on the left; a g closes a loop"),
    's>g': ('run_d2@M*1.5 end_any@Bl loop*-1.5', 0, "an s's spine runs down to the right and its lower curve ends on the left; a g closes a loop"),
    'g>y': ('loop*1.5 bay_U@A*-2', 0, 'a g closes its head; a y opens upward between its arms'),
    # H
    'H>J': ('junction@M*1.5 hx2@D bay_L@Mc*-1', 0, "an H's bar joins its stems and its lower rows cross both legs; a J hooks round with an opening on the left"),
    'h>J': ('junction@M*1.5 hx2@D bay_D@B bay_L@Mc*-1 end_left@l*-1.5 ax2@c*-1.5 ax1@B*1.5 bay_D@A', 0, "an h's arch joins its stem and opens downward between its legs, and the / lines at the bottom cross it once; a J hooks back with an opening on the left and an end pointing left"),
    'j>h': ('ax2@c*1.5 bay_D@A*-2 ax1@B*-1', 0, "a j's hook curls back; an h opens downward between stem and leg"),
    'H>K': ('edge_r@r*1.5 bay_R@M*-1.5 bay_R@A*-1.5 junction@r*1.5 run_v@r', 0, 'an H has a second upright down the right, which its bar meets; a K opens to the right between its arms'),
    'H>k': ('edge_r@r*1.5 bay_R@A*-1.5 junction@r*1.5 run_v@r', 0, 'an H has a second upright down the right, which its bar meets; a k opens to the right between its arms'),
    'h>k': ('edge_r@r*1.5 run_v@r bay_R@M*-1.5 crossing@M*-1.5 vx2@r*-1 end_ur@r*-1 bay_D@A', 0, "an h's leg comes straight down the right and its arch opens downward; a k opens to the right between its arms, which cross or meet its stem"),
    'H>l': ('hx2@D*1.5 hx2@B vx1@r*-1', 0, "an H's lower rows cross both stems; an l is one stroke"),
    'h>l': ('hx2@D*1.5 hx2@B vx1@r*-1', 0, "an h's lower rows cross its stem and its leg; an l is one stroke"),
    'H>M': ('hx2@M*1.5 hx3@M*-1.5 hx3@A*-1 bay_U@Tc*1.5', 0, "an H's middle rows cross two uprights and it opens upward between them; an M's middle rows cross three or more strokes"),
    'm>H': ('hx3@M*1.5 hx3@A bay_U@Tc*-1.5', 0, "an m's middle rows cross three legs; an H opens upward between its uprights"),
    'm>h': ('hx3@M*1.5 hx1@U*-1.5 run_v@c*-1', 0, "an m's middle rows cross three legs; an h's upper rows cross only its tall stem"),
    'H>N': ('junction@r*1.5 hx3@M*-1.5 hx3@A*-1', 0, "an H's bar meets its right upright in a junction and its rows cross only the two uprights; an N's diagonal makes its middle rows cross three strokes"),
    'h>n': ('hx1@T ink@Tr*-1.5 edge_t@T*-1.5', 0, "an h's top is its stem alone, with nothing at the top right and no top edge; an n's arch spans the top"),
    'n>H': ('edge_t@T*1.5 bay_U@A*-2', 0, "an n's top is an arch with a flat top edge; an H opens upward between its uprights"),
    'H>u': ('bay_D@A*1.5 edge_b@B*-1.5 run_h@B*-1 end_down@B*1.5 bay_D@B*1.5', -0.5, "an H opens downward between its legs, which end pointing down; a u's sides join in a curve along the bottom"),
    'U>H': ('end_down@B*-1.5 bay_D@B*-1.5', 0.5, "a U's sides join along the bottom; an H's legs end pointing down and open downward"),
    'H>v': ('bay_D@A*1.5 junction@M hx1@B*-1', 0, "an H's legs open downward under its bar; a v's arms meet in one point at the bottom"),
    'W>H': ('bay_U@Mr*1.5 run_v@c*-1.5', 0, "a W opens upward on its right; an H's stems are straight"),
    'h>X': ('hx1@T*1.5 hx2@T*-1.5 bay_U@T*-1.5', 0, "an h's top rows cross only its tall stem; an X's top rows cross both strokes and open upward"),
    'h>x': ('hx1@T*1.5 end_any@Tl*-1 end_ul@T*-1.5', 0, "an h's top is its stem alone; an x's top-left stroke ends pointing up-left"),
    'y>H': ('hx1@B*1.5 bay_D@A*-2', 0, "a y's tail is one stroke at the bottom; an H opens downward between its legs"),
    # I
    'J>I': ('bay_L@B*1.5 end_any@Bl hx2@D end_left@l end_any@Br*-1 edge_b@B*-1 end_dr@B*-1 ax1@B*-0.5', 0, "a J's foot hooks to the left and ends pointing left, leaving an opening on the left at the bottom; an I ends straight down, or with a bar both ways"),
    'j>i': ('end_left@l*1.5 end_down@B*-1.5', 0, "a j's hook ends pointing left; an i's foot ends pointing down"),
    'L>I': ('run_h@B*1.5 bay_R@M end_any@Br*0.5 edge_t@T*-1 run_h@T*-1', -0.75, 'an L has a foot to the right and nothing across its top; an I is a stick or has a bar across its top'),
    'l>i': ('gap*-2', 0.3, 'an l has no dot; an i has one'),
    'I>r': ('narrow*1.5 wide_r*-1.5 run_h@A*-1', 0, "an I (or i) stays narrow; an r's arm reaches out to the right"),
    'i>r': ('narrow*1.5 wide_r*-1.5 run_h@A*-1', 0, "an I (or i) stays narrow; an r's arm reaches out to the right"),
    'T>I': ('run_h@T*1.5 end_any@Tl end_any@Tr run_h@B*-1.5 edge_b@B*-1', 0, 'a T has a bar across the top only; an I is a stick or has bars at both ends'),
    't>I': ('junction@M*1.5 run_h@B*-1.5', 0, "a t's bar crosses its stem part way down; an I is a stick or has bars at both ends"),
    'I>x': ('hx2@A*-2 run_d1@A*-1.5', 0.5, "an I is one upright stroke; an x's strokes cross through its middle"),
    'I>Y': ('bay_U@A*-1.5 hx2@U*-1.5 hx1@T', 0, "an I's top is one stroke; a Y's upper rows cross both arms and it opens upward"),
    'i>Y': ('hx1@T*1.5 hx2@U*-1.5 bay_U@A*-1.5', 0, "an I's (or i's) top rows cross one stroke; a Y's top rows cross both arms and open upward"),
    'I>Z': ('bx1@r run_d1@M*-1.5 vx3@c*-1', 0.3, "an I is upright; a Z's diagonal runs across its middle and its centre columns cross three strokes"),
    'I>z': ('bx1@r run_d1@M*-1.5 vx3@c*-1', 0.3, "an I is upright; a z's diagonal runs across its middle"),
    # J
    'J>L': ('ax2@c*1.5 ax1@A*-1.5', 0, "a J's hook curls back up, so the / lines through its middle cross two strokes; an L's foot runs straight out"),
    'J>l': ('ax2@c*1.5 marg_t@r*-1 ax1@B*-1', 0, "a J's hook comes back, so the / lines through its centre cross two strokes; an l is one stroke"),
    'j>l': ('end_left@l*1.5 end_down@B*-1', 0, "a j's hook ends pointing left; an l ends pointing down"),
    'J>R': ('loop_hi*-2 stem_lh@A*-1', 0.5, "a J has no bowl; an R's bowl sits right of its stem"),
    'J>S': ('vx1@r*1.5 junction@T edge_r@T vx3@c*-1.5 run_d2@M*-1 edge_b@Bc*-1 bay_R@T*-1', 0, "a J's right columns cross only its stem and its bar meets the stem; an S's centre columns cross three strokes and its middle runs down to the right"),
    'J>s': ('vx1@r*1.5 junction@T edge_r@T vx3@c*-1.5 run_d2@M*-1 edge_b@Bc*-1 bay_R@T*-1 S_body*-1.5 end_dl@A*-1', 0, "a J's right columns cross only its stem and its bar meets the stem; an s's centre columns cross three strokes and its spine runs down to the right"),
    'J>T': ('hx2@D*1.5 hx2@B bay_L@M junction@M*-1.5 hx1@B*-1 end_down@B*-1.5 end_dl@A*-1', 0, "a J hooks left at the bottom, so it opens to the left and its lower rows cross two strokes; a T's bar meets its stem in a junction and the stem ends alone, pointing down"),
    'T>j': ('run_h@T*1.5 gap*-1.5 bay_L@B*-1', 0, "a T's top is a bar; a j's top is a dot and its tail hooks left"),
    't>j': ('crossing@M*1.5 gap*-1.5', 0, "a t's bar crosses its stem; a j's top is a separate dot"),
    'J>V': ('hx1@U*1.5 hx2@U*-1.5 bay_U@T*-1.5', 0, "a J's (or j's) upper rows cross one stroke; a V's cross both arms and open upward"),
    'j>v': ('hx1@U*1.5 hx2@U*-1.5 bay_U@T*-1.5', 0, "a J's (or j's) upper rows cross one stroke; a V's cross both arms and open upward"),
    'J>x': ('bay_L@M*1.5 hx2@D junction@Mc*-2', 0, "a J opens to the left above its hook and its lower rows cross two strokes; an x's strokes cross in the centre, which its centre columns cross once"),
    'J>Y': ('bay_L@Mc*1.5 hx2@D edge_t@T bay_U@T*-1.5', 0, 'a J hooks back left at the bottom under a bar; a Y opens upward between its arms'),
    'J>Z': ('bay_L@B*1.5 run_d1@M*-1.5', 0, "a J's hook opens to the left at the bottom; a Z's diagonal runs across its middle"),
    'J>z': ('bay_L@B*1.5 run_d1@M*-1.5', 0, "a J's hook opens to the left at the bottom; a z's diagonal runs across its middle"),
    # K
    'L>K': ('hx1@T*1.5 hx2@U*-1.5 junction@A*-1.5', 0, "an L's top rows cross its stem alone and nothing joins it; a K's arms meet its stem"),
    'L>k': ('hx1@T*1.5 hx2@U*-1.5 junction@A*-1.5', 0, "an L's top rows cross its stem alone and nothing joins it; a K's arms meet its stem"),
    'R>k': ('hole@A*1.5 bay_U@A*-1', 0, 'an R closes a bowl at the top; a k opens upward between stem and arm'),
    'K>t': ('hx2@A edge_t@M*-1.5', 0, "a K's rows cross stem and arm; a t's bar has nothing above it"),
    'k>t': ('hx2@A*1.5 end_dr@B hx2@T ax2@c*-1.5 edge_t@Mc*-1.5', 0, "a k's rows cross stem and arm, and its leg ends at the bottom right; a t's bar has a top edge in the middle and crosses its stem once"),
    'v>K': ('bay_U@M*1.5 junction@A*-2 bay_R@A*-1.5', 0, "a v opens upward with no joins; a K's arms meet its stem and open to the right"),
    'w>k': ('bay_R@A*-2 vx2@r*-1', 0.5, 'a w does not open to the right; a k opens to the right between its arms'),
    'K>X': ('junction@l bay_L@M*-1.5 vx2@l*-1', 0, "a K's arms meet its stem on the left; an X opens to the left as well as to the right"),
    'K>x': ('junction@l*1.5 run_v@l bay_L@A*-1.5 vx2@l*-1', 0, "a K's arms meet a straight stem on the left; an x opens to the left as well, and its left columns cross two strokes"),
    'k>x': ('junction@l*1.5 run_v@l bay_L@A*-1.5 vx2@l*-1 end_ul@T*-1.5 end_any@Tl*-1', 0, "a k's arms meet a straight stem on the left; an x opens to the left as well, and its top-left stroke ends pointing up-left"),
    # L
    'l>M': ('narrow*2 wide_r*-1 wide_l*-1', 0, 'an l stays narrow; an M spreads across the width'),
    'l>r': ('narrow*1.5 edge_t@T*-1.5 end_any@M*-1', 0, "an l is one narrow stroke; an r's arm starts at the top and ends to the right"),
    'T>l': ('run_h@T*1.5 end_any@Tl end_any@Tr run_h@A*2 edge_t@T*1.5', -1.0, 'a T has a bar across its top; an l is one stroke'),
    't>l': ('junction@M*1.5 end_any@Ml end_any@Mr end_dl@B*-1.5', -0.25, "a t's bar crosses its stem and ends on both sides; an l ends pointing down-left at the bottom"),
    'Y>l': ('hx2@U*1.5 edge_l@T*-1 edge_r@T*-1', 0, "a Y's upper rows cross both arms; an l is one stroke"),
    'y>l': ('hx2@U*1.5 edge_l@T*-1 edge_r@T*-1', 0, "a y's upper rows cross both arms; an l is one stroke"),
    'z>L': ('edge_t@T*1.5 end_up@T*-1.5', 0, "a z has a bar across its top; an L's stem rises to an end pointing up"),
    # M
    'M>N': ('ax1@B*1.5 bx1@T*-1 bay_U@M*-1', 0, "an M's legs are crossed once by the / lines at the bottom; an N's diagonal and right stem open upward on the right"),
    'M>n': ('hx3@D*1.5 hx3@M hx2@D*-1 hx2@M*-1', 0, "an M's lower rows cross three or four strokes; an n's cross two"),
    'm>N': ('ax1@B*1.5 bx1@T*-1 bay_U@M*-1', 0, "an M's legs are crossed once by the / lines at the bottom; an N's diagonal and right stem open upward on the right"),
    'm>n': ('hx3@M*1.5 hx3@A hx2@D*-1.5', 0, "an m's middle rows cross three legs; an n's cross two"),
    'M>T': ('bay_D@A*1.5 hx3@M hx1@M*-1.5 edge_l@c*-1', 0, "an M opens downward between its legs and its middle rows cross three strokes; a T's stem stands alone in the centre"),
    'T>m': ('hx1@M*1.5 hx3@A*-1.5 bay_D@A*-1.5', 0, "a T's middle rows cross its stem alone; an m's cross three legs"),
    'M>W': ('bay_D@M*1.5 bay_D@Mc bay_U@r*-1.5 bay_U@M*-1', 0, 'an M opens downward between its legs, up into the middle; a W opens upward, in the middle and on the right'),
    'M>w': ('bay_D@M*1.5 bay_U@r*-1.5 bay_U@M*-1', 0, 'an M (or m) opens downward between its legs; a w opens upward, on the right and in the middle'),
    'm>W': ('bay_D@M*1.5 bay_D@Mc bay_U@r*-1.5 bay_U@M*-1', 0, 'an M opens downward between its legs, up into the middle; a W opens upward, in the middle and on the right'),
    'm>w': ('bay_D@M*1.5 bay_U@r*-1.5 bay_U@M*-1', 0, 'an M (or m) opens downward between its legs; a w opens upward, on the right and in the middle'),
    # N
    'r>n': ('hx1@B*1.5 bay_D@B*-1.5 hx2@B*-1', 0, "an r's bottom is its stem alone; an n opens downward between its legs"),
    'N>V': ('wide_l end_any@Bl hx1@B*-1.5', 0, "an N's left stem reaches down to its own end at the bottom left; a V's arms meet in one point"),
    'n>V': ('run_h@M*1.5 edge_b@M bay_U@Mc*-1', 0, "an n's arch runs across the middle with space under it; a V opens upward in the middle"),
    'N>W': ('bx1@B ax1@T*-1', 0, "an N's \\ lines at the bottom cross one stroke; a W's / lines at the top cross one arm"),
    'N>w': ('bx1@B ax1@T*-1 edge_b@B*-1 bay_D@A*1.5 bay_D@l', 0, 'an N opens downward and its \\ lines at the bottom cross one stroke; a w joins along the bottom'),
    'n>w': ('bay_D@B*1.5 ink@Tc bay_U@T*-1.5', 0, 'an m or n opens downward under its arches; a w opens upward between its arms'),
    'N>X': ('bay_D@A bay_R@A*-1.5 hx1@M*-1.5', 0.3, "an N's middle rows cross two or three strokes; an X's cross only where its strokes cross"),
    'N>x': ('bay_D@A bay_R@A*-1.5 hx1@M*-1.5', 0.3, "an N's (or n's) middle rows cross two or three strokes and it opens downward; an x's middle rows cross only its crossing point and it opens to the side"),
    'n>X': ('bay_D@A bay_R@A*-1.5 hx1@M*-1.5', 0.3, "an n's middle rows cross both legs; an X's cross only where its strokes cross"),
    'n>x': ('bay_D@A bay_R@A*-1.5 hx1@M*-1.5', 0.3, "an n's middle rows cross both legs and it opens downward; an x's middle rows cross only its crossing point"),
    # O
    'O>p': ('run_h@B*1.5 junction@A*-1.5 end_down@A*-1', 0, "an O closes along its bottom; a p's stem meets its bowl and ends pointing down"),
    'Q>O': ('junction@A*1.5 end_any@Br end_any@B*0.5', -0.25, "a Q's tail meets or crosses its bowl and ends at the bottom right; an O is one closed curve"),
    'Q>o': ('junction@A*1.5 end_any@Br end_any@B*0.5', -0.25, "a Q's tail meets or crosses its bowl; an o is one closed curve"),
    # P
    'P>q': ('bx1@B edge_r@r*-1.5 edge_b@A*-1 desc_r@A*-1.5 ax1@B*-1 hole@Mr*1.5', 0, "a P's bowl sits right of its stem; a q's stem runs down the right and continues below its bowl"),
    'p>q': ('bx1@B edge_l@l hole@r edge_r@r*-1.5 edge_b@A*-1 hole@l*-1.5 edge_r@Br*-1', 0, "a p's stem is on the left; a q's stem runs down the right"),
    'R>P': ('hx2@B*1.5 bay_R@A hx1@B*-1.5', 0, "an R's bottom rows cross its stem and its leg, and it opens to the right between bowl and leg; a P's bottom is its stem alone"),
    'R>p': ('hx2@B*1.5 bay_R@A hx1@B*-1.5', 0, "an R's bottom rows cross its stem and its leg, and it opens to the right between bowl and leg; a P's bottom is its stem alone"),
    # Q
    'q>R': ('desc_r@B*1.5 hx1@B edge_l@Ml*-1.5 hx2@B*-1', 0, "a q's stem continues alone below its bowl; an R's leg and stem make two strokes at the bottom"),
    't>q': ('loop*-2 stem_rh@A*-1', 0.5, 'a t has no loop; a q closes a bowl beside its stem'),
    # R
    'R>T': ('edge_l@Ml*1.5 run_v@l end_dr@A*0.5', 0, "an R has a straight stem down the left side; a T's stem stands in the centre"),
    'r>T': ('edge_l@Ml*1.5 run_v@l end_dr@A*0.5 end_any@Tl*-1.5 run_h@T*-1 edge_l@l', 0, "an r's stem is its left edge and its arm ends at the right; a T's bar runs out to the left of its stem and ends there"),
    't>r': ('end_any@Tc*1.5 end_any@Ml edge_l@l*-1', 0, "a t's stem rises above the bar and the bar runs out to the left; an r's stem is its left edge"),
    'V>r': ('end_any@Mr*-1.5 edge_b@r*-1.5 end_dr@M*-1', 0.5, "a V's arms end at the top; an r's arm ends at the middle right with nothing under it"),
    'v>r': ('bay_U@M end_any@Mr*-1.5 end_dr@A*-1', 0, "a v opens upward between its arms; an r's arm ends at the middle right"),
    'r>X': ('hx1@B*1.5 hx2@D*-1.5 bay_D@B*-1', 0, "an r's bottom is its stem alone; an X's lower strokes spread apart"),
    'r>Y': ('end_any@M*1.5 edge_b@A marg_r@T*-1', 0, "an r's arm ends in the middle rows with nothing under it; a Y's arms reach out to the top right"),
    # S
    'S>X': ('vx3@c*1.5 vx1@c*-2 junction@Mc*-1', 0, "an S's centre columns cross three strokes (top, spine, bottom); an X's centre columns cross once, where its strokes cross"),
    's>x': ('vx3@c*1.5 vx1@c*-2 junction@Mc*-1', 0, "an s's centre columns cross three strokes (top, spine, bottom); an x's centre columns cross once, where its strokes cross"),
    's>y': ('run_h@B*1.5 edge_t@T bay_U@T*-1.5', 0, "an s's top and bottom are curves across its width; a y opens upward between its arms"),
    'S>Z': ('S_body*1.5 run_d2@M bay_L@T*-1.5 run_d1@M*-1.5', 0, "an S's spine runs down to the right; a Z's diagonal runs down to the left and it opens to the left under its top bar"),
    'S>z': ('S_body*1.5 run_d2@M bay_L@T*-1.5 run_d1@M*-1.5', 0, "an S's spine runs down to the right; a z's diagonal runs down to the left and it opens to the left under its top bar"),
    's>Z': ('S_body*1.5 run_d2@M bay_L@T*-1.5 run_d1@M*-1.5', 0, "an s's spine runs down to the right; a Z's diagonal runs down to the left and it opens to the left under its top bar"),
    's>z': ('S_body*1.5 run_d2@M bay_L@T*-1.5 run_d1@M*-1.5', 0, "an s's spine runs down to the right; a z's diagonal runs down to the left and it opens to the left under its top bar"),
    # T
    'T>x': ('end_left@A*1.5 hx2@A*-1.5', 0, "a T's bar ends pointing left at its left end; an x's rows cross both strokes"),
    't>X': ('hx1@T*1.5 hx2@T*-1.5 bay_U@A*-1.5', 0, "a t's top is its stem alone; an X's top rows cross both strokes and open upward"),
    't>x': ('hx1@B hx2@A*-1.5 end_dr@B*-1.5', 0, "a t's bottom is its stem alone; an x's rows cross both strokes"),
    'T>Y': ('edge_t@T*1.5 run_h@T bay_U@T*-1.5 bay_U@M*-1', 0, "a T's top is one bar with nothing above it; a Y opens upward between its arms"),
    'T>y': ('edge_t@T*1.5 run_h@A hx2@T*-1.5', 0, "a T's top is one bar; a y's top rows cross both arms"),
    't>Y': ('crossing@M*1.5 bay_U@T*-1.5', 0, "a t's bar crosses its stem; a Y opens upward between its arms"),
    'T>Z': ('edge_b@B*-2 bay_R@A*-1.5', 0.5, 'a T has no base and does not open to the right; a Z runs along a base'),
    'T>z': ('end_any@Bc end_any@B run_h@B*-1.5 edge_b@B*-1', 0, "a T's stem ends alone at the bottom; a z runs along a base"),
    'Z>t': ('bay_L@M*1.5 bx1@T*-1', 0, "a Z opens to the left under its top bar; a t's stem rises alone above its bar"),
    't>z': ('junction@M end_any@Bc run_h@B*-1.5 edge_b@B*-1', 0, "a t's bar crosses its stem and the stem ends alone at the bottom; a z runs along a base"),
    # U
    'U>V': ('bx2@M*1.5 bx1@B*-1 end_any@B*-1 edge_b@B*1.5 run_h@B*1.5', -0.5, "a U's sides join in a flat curve along the bottom and its \\ lines through the middle cross both sides; a V's arms meet in a point"),
    'U>v': ('run_h@B*0.5 edge_b@B*0.5 hx1@B*-1 edge_l@Bc*-1.5 vx2@A*1.5', 0, "a U's columns cross both its sides and its bottom; a v's arms meet in a point at the bottom centre"),
    'u>V': ('vx2@A run_h@B*0.5 edge_b@B*0.5 hx1@B*-1', 0, "a U's sides come down to a round bottom, so its columns cross two strokes and its bottom runs flat; a v's arms meet in a point"),
    'u>v': ('wide_l*1.5 edge_l@c*-1 run_v@c*-1 end_ul@l*-1.5 bx1@B*-1 bx2@T', 0, "a u's left side reaches the left edge and the \\ lines at its top cross both sides; a v's left arm ends pointing up-left and it is crossed once at the bottom"),
    'U>W': ('hx2@U*1.5 hx3@U*-1.5 hx3@M*-1.5', -0.5, "a U's (or u's) upper rows cross its two sides; a w's cross three strokes"),
    'U>w': ('hx2@U*1.5 hx3@U*-1.5 hx3@M*-1.5', -0.5, "a U's (or u's) upper rows cross its two sides; a w's cross three strokes"),
    'u>W': ('hx2@U*1.5 hx3@U*-1.5 hx3@M*-1.5 hx3@A*-1 bay_U@Mr*-1 edge_r@c', -0.5, "a u's upper rows cross its two sides; a W's cross three or more strokes and it opens upward on the right"),
    'u>w': ('hx2@U*1.5 hx3@U*-1.5 hx3@M*-1.5', -0.5, "a U's (or u's) upper rows cross its two sides; a w's cross three strokes"),
    'U>X': ('bx2@c*1.5 bay_U@M hx1@M*-1.5', 0, "a U's middle rows cross both sides and it opens upward; an X's middle rows cross only its crossing point"),
    'u>x': ('bx2@c*1.5 bay_U@M hx1@M*-1.5', 0, "a u's middle rows cross both sides; an x's cross only its crossing point"),
    'y>u': ('tail_below_r hx1@D*1.5 edge_b@B*-1.5', 0, "a y's tail hangs below as one stroke; a u's sides join along the bottom"),
    # V
    'V>Y': ('hx2@M hx1@M*-1.5 junction@c*-1 edge_l@Ml*1.5 hx2@D', 0.25, "a V's left arm runs down the left, so its middle and lower rows cross both arms; a Y's arms meet its stem in a junction and its middle rows cross the stem alone"),
    'V>y': ('run_v@Ml*1.5 junction@c*-1 edge_l@Ml*1.5 run_v@l', 0.25, "a V's left arm runs straight down the left; a y's short arm meets its long stroke in a junction"),
    'v>Y': ('edge_l@l*1.5 ink@Ml hx1@M*-1.5', 0, "a v's left arm runs down the left; a Y's middle rows cross its stem alone"),
    # X
    'X>y': ('hx2@D*1.5 end_dr@A hx1@B*-1', 0, "an X's lower strokes spread apart, so its lower rows cross two; a y's tail is one stroke"),
    'x>Y': ('edge_b@A*1.5 end_dr@B hx2@B', 0, "an x's lower strokes spread apart and end at the bottom right too; a Y's stem stands alone at the bottom"),
    'x>y': ('hx2@D*1.5 end_dr@A hx1@B*-1', 0, "an x's lower strokes spread apart; a y's tail is one stroke"),
    'X>Z': ('junction@M*1.5 hx2@U crossing@M vx3@A*-1.5 bay_L@M*-1', 0, "an X's strokes cross in the middle and its upper rows cross both; a Z's columns cross three strokes and it opens to the left"),
    'z>x': ('run_h@Tc*1.5 hx2@A*-1.5 vx1@c*-1', 0, "a z runs a bar across its top and its rows cross one stroke at a time; an x's rows cross both strokes"),
}


def _merge_referees(refs):
    """Two referees written for one pair ("x>y" and "y>x") become one: the second one's evidence is turned round
    (its for-terms count against x) and both reasons are kept."""
    out = {}
    for key, (terms, bias, why) in refs.items():
        x, y = key.split(">")
        if f"{y}>{x}" in out:
            t0, b0, w0 = out[f"{y}>{x}"]
            flipped = " ".join(f"{n}*{-w:g}" for n, w in parse_terms(terms))
            out[f"{y}>{x}"] = (t0 + " " + flipped, b0 - bias, w0 + "; also: " + why)
        elif key in out:
            t0, b0, w0 = out[key]
            out[key] = (t0 + " " + terms, b0 + bias, w0 + "; also: " + why)
        else:
            out[key] = (terms, bias, why)
    return out


def parse_terms(s):
    out = []
    for tok in s.split():
        name, _, w = tok.partition("*")
        out.append((name, float(w) if w else 1.0))
    return out


def style_weights(rule):
    """Positive terms (normalised), contradictions, and the closed-world penalty for everything else salient."""
    pos, _, neg = rule.partition("|")
    toks = pos.split()
    opt = [t[1:] for t in toks if t.startswith("?")]
    P = parse_terms(" ".join(t for t in toks if not t.startswith("?")))
    N = parse_terms(neg)
    tot = sum(w for _, w in P)
    mentioned = {n for n, _ in P} | {n for n, _ in N} | set(opt)
    for n in list(mentioned):
        mentioned |= set(IMPLIES.get(n, []))
    out = [(n, w / tot) for n, w in P] + [(n, -NEG * w) for n, w in N]
    out += [(s, -UNEXP) for s in SALIENT if s not in mentioned]
    return out


def style_bias(rule):
    """Tie-break: when two styles both score 1.0, the one that explains more of the drawing wins."""
    pos = [t for t in rule.partition("|")[0].split() if not t.startswith("?")]
    return round(0.01 * len(pos), 4)


REFEREES = _merge_referees(REFEREES)
