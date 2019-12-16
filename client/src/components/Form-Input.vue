<template>
    <input 
        class='form-input'
        type='text'
        :placeholder='placeholderText'
    >
</template>

<script>
    export default {
        name: 'FormInput',
        props: {
            placeholderText: String,
        },
    }
</script>

<style scoped>
    .form-input {
        border: none;
        height: 50px;
        width: 350px;
        border-radius: 50px;
        background-color: #D7D7D7;
        padding-left: 20px;
        outline: none;
        -webkit-box-sizing: border-box;
        -moz-box-sizing: border-box;
        box-sizing: border-box;
    }
</style>