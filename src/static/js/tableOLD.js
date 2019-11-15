const $tableID = $('#table');

 const newTr = ``;
/*
 const newTr = `
<tr class="hide">
	<td class="pt-3-half" contenteditable="true">${document.getElementById("idno").innerHTML}</td>
	<td class="pt-3-half" contenteditable="true">${document.getElementById("pans").innerHTML}</td>
	<td class="pt-3-half" contenteditable="true">${document.getElementById("ans").innerHTML}</td>
	<td class="pt-3-half" contenteditable="true">${defcmarks}</td>
	<td class="pt-3-half" contenteditable="true">${defwmarks}</td>
	<td class="pt-3-half" contenteditable="true">${defsub}</td>

	<td>
		<span class="table-remove"><button type="button" class="btn btn-danger btn-rounded btn-sm my-0 waves-effect waves-light">Remove</button></span>
	</td>
</tr>`; */
// Below code is for Add More...
$('.table-add1').on('click', 'i', () => {
	console.log("clcickk");
		var rowno = document.getElementById('inputno').value;
		var defcmarks = document.getElementById('def-cmarks').value ;
		var defwmarks = document.getElementById('def-wmarks').value ;
		var defsub = document.getElementById('def-sub').value ;
		const newTr = `
		<tr class="hide">
			<td class="pt-3-half" contenteditable="true"></td>
			<td class="pt-3-half" contenteditable="true">4</td>
			<td class="pt-3-half" contenteditable="true">A</td>
			<td class="pt-3-half" contenteditable="true">${defcmarks}</td>
			<td class="pt-3-half" contenteditable="true">${defwmarks}</td>
			<td class="pt-3-half" contenteditable="true">${defsub}</td>

			<td>
				<span class="table-remove"><button type="button" class="btn btn-danger btn-rounded btn-sm my-0 waves-effect waves-light">Remove</button></span>
			</td>
		</tr>`;
		/*document.getElementById("def-cmarkss").innerHTML = defcmarks;
		document.getElementById("def-wmarkss").innerHTML = defwmarks;
		document.getElementById("def-subs").innerHTML = defsub;*/


/*    console.log(defcmarks);
		for (var i = 1; i < rowno; i++) {

		const $clone = $tableID.find('tbody tr').last().clone(true).removeClass('hide table-line');

		if ($tableID.find('tbody tr').length === 0) {

			$('tbody').append(newTr);
		}
}
		$tableID.find('table').append($clone);
*/
var rowno = document.getElementById('inputno').value;
for (var i = 0; i < rowno; i++) {



			$('tbody').append(newTr);

}
	});





// Here code is for Add (Single row)
 $('.table-add').on('click', 'i', () => {
		/***var rowno = document.getElementById('inputno').value;
		var defcmarks = document.getElementById('def-cmarks').value ;
		var defwmarks = document.getElementById('def-wmarks').value ;
		var defsub = document.getElementById('def-sub').value ;

		document.getElementById("def-cmarkss").innerHTML = defcmarks;
		document.getElementById("def-wmarkss").innerHTML = defwmarks;
		document.getElementById("def-subs").innerHTML = defsub; ***/
		var defcmarks = document.getElementById('def-cmarks').value ;
		var defwmarks = document.getElementById('def-wmarks').value ;
		var defsub = document.getElementById('def-sub').value ;
		const newTr = `
		<tr class="hide">
			<td class="pt-3-half" contenteditable="true"></td>
			<td class="pt-3-half" contenteditable="true">4</td>
			<td class="pt-3-half" contenteditable="true">A</td>
			<td class="pt-3-half" contenteditable="true">${defcmarks}</td>
			<td class="pt-3-half" contenteditable="true">${defwmarks}</td>
			<td class="pt-3-half" contenteditable="true">${defsub}</td>

			<td>
				<span class="table-remove"><button type="button" class="btn btn-danger btn-rounded btn-sm my-0 waves-effect waves-light">Remove</button></span>
			</td>
		</tr>`;

	 const $clone = $tableID.find('tbody tr').last().clone(true).removeClass('hide table-line');

	if ($tableID.find('tbody tr').length === 0) {

		 $('tbody').append(newTr);
	 }

	 $tableID.find('table').append($clone);


 });


 // Remove row function
 $tableID.on('click', '.table-remove', function () {

	 $(this).parents('tr').detach();
 });
