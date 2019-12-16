<template>
    <div class='form-title'>
        {{ message }}
    </div>
</template>

<script>
export default {
    name: 'FormTitle',
    props: {
        message: String,
    },
}
</script>

<style scoped>
    .form-title {
        font-family: 'arial';
        font-size: 64px;
        color: white;
    }
</style>