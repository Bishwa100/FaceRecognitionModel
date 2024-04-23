from flask import Flask, request
from flask_restx import Api, Resource, fields
from flask import Flask, request
from flask_restx import Api, Resource, fields

app = Flask(__name__)

# Model for the data
data_model = api.model('Data', {
	'id': fields.Integer(readOnly=True, description='The unique identifier'),
	'name': fields.String(required=True, description='The name of the data'),
	'value': fields.String(required=True, description='The value of the data'),
})

# Sample data
sample_data = [
	{'id': 1, 'name': 'Data 1', 'value': 'Value 1'},
	{'id': 2, 'name': 'Data 2', 'value': 'Value 2'},
]

@api.route('/data')
class DataList(Resource):
	@api.doc('list_data')
	def get(self):
		"""List all data"""
		return sample_data

	@api.doc('create_data')
	@api.expect(data_model)
	def post(self):
		"""Create a new data"""
		new_data = request.json
		new_data['id'] = len(sample_data) + 1
		sample_data.append(new_data)
		return new_data, 201

@api.route('/data/<int:data_id>')
@api.response(404, 'Data not found')
@api.param('data_id', 'The data identifier')
class Data(Resource):
	@api.doc('get_data')
	def get(self, data_id):
		"""Get data by ID"""
		data = next((item for item in sample_data if item['id'] == data_id), None)
		if data is None:
			api.abort(404, "Data not found")
		return data

	@api.doc('update_data')
	@api.expect(data_model)
	def put(self, data_id):
		"""Update data by ID"""
		data = next((item for item in sample_data if item['id'] == data_id), None)
		if data is None:
			api.abort(404, "Data not found")
		updated_data = request.json
		data.update(updated_data)
		return data

	@api.doc('delete_data')
	def delete(self, data_id):
		"""Delete data by ID"""
		global sample_data
		sample_data = [item for item in sample_data if item['id'] != data_id]
		return {'message': 'Data deleted successfully'}, 200

if __name__ == '__main__':
	app.run(debug=True)
